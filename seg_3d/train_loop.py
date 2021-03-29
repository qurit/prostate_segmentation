import os
import json
import random
import logging
import numpy as np
from time import time

from seg_3d import losses
from seg_3d.data.dataset import ImageToImage3D
from seg_3d.config import get_cfg
import seg_3d.modeling.backbone.unet
import seg_3d.modeling.meta_arch.segnet

import torch
from torch.utils.data import DataLoader
from detectron2.utils.file_io import PathManager
from detectron2.utils.collect_env import collect_env_info
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, EventStorage
from detectron2.utils.logger import setup_logger


logger = logging.getLogger("detectron2")


# TODO:
# - resume option
# - loads weights
# - default directory
# - implementation for eval
# - plot loss
# - use wandb and sacred
def setup_config():
    cfg = get_cfg()

    cfg.MODEL.DEVICE = "cuda:0"
    cfg.SEED = 99

    # pipeline modes
    cfg.EVAL_ONLY = False
    cfg.TEST.EVAL_PERIOD = 0  # The period (in terms of steps) to evaluate the model during training. Set to 0 to disable
    cfg.EARLY_STOPPING.ENABLE = False
    cfg.EARLY_STOPPING.PATIENCE = 10
    cfg.EARLY_STOPPING.MONITOR = "val_loss"

    # paths
    cfg.DATASET_PATH = "/home/yous/Desktop/ryt/image_dataset"
    cfg.OUTPUT_DIR = "seg_3d/output/test-1"

    # dataset options
    cfg.MODALITY = "CT"
    cfg.ROIS = ["Bladder"]
    # TODO: centre crop + other preprocessing options here

    # model architecture
    cfg.MODEL.META_ARCHITECTURE = "SemanticSegNet"
    cfg.MODEL.BACKBONE.NAME = "UNet3D"
    # specify UNet params which are defined in Abstract3DUNet
    cfg.UNET.in_channels = 1
    cfg.UNET.out_channels = 1
    cfg.UNET.f_maps = 8

    # loss
    cfg.LOSS.FN = "BCEDiceLoss"  # available loss functions are inside losses.py
    # specify loss params
    cfg.LOSS.PARAMS.alpha = 1.0
    cfg.LOSS.PARAMS.beta = 0.0

    # solver params
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 10  # Save a checkpoint after every this number of iterations
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (30000,)  # The iteration number to decrease learning rate by GAMMA

    return cfg


def get_es_result(mode, current, best_so_far):
    """Returns true if monitored metric has been improved"""
    if mode == 'max':
        return current > best_so_far
    elif mode == 'min':
        return current < best_so_far


# TODO:
# - switch from default optim
# - evaluate on val set
# - could load up all scans into memory
def train(cfg, model):
    model.train()

    # get training dataset
    dataset = ImageToImage3D(dataset_path=cfg.DATASET_PATH, modality=cfg.MODALITY, rois=cfg.ROIS)

    # get default optimizer (torch.optim.SGD) and scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # init loss criterion
    loss = losses.get_loss_criterion(cfg)(**cfg.LOSS.PARAMS)
    logger.info("Loss:\n{}".format(loss))

    # init checkpointers
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    # init writers which periodically output/save metric scores
    writers = [CommonMetricPrinter(max_iter),
               JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
               TensorboardXWriter(cfg.OUTPUT_DIR)]

    # init early stopping
    best_monitor_metric = None
    es_count = 0

    # measuring the time elapsed
    train_start = time()
    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        # start main training loop
        for iteration, batched_inputs in zip(range(start_iter, max_iter),
                                             DataLoader(dataset, batch_size=1, shuffle=False)):

            storage.step()
            sample = batched_inputs["image"].unsqueeze(0).float()
            labels = batched_inputs["gt_mask"].float().to(cfg.MODEL.DEVICE)

            # do a forward pass, input is of shape (b, c, d, h, w)
            preds = model(sample).squeeze(0)

            optimizer.zero_grad()
            training_loss = loss(preds, labels)  # https://github.com/wolny/pytorch-3dunet#training-tips
            print(training_loss)
            training_loss.backward()
            optimizer.step()

            storage.put_scalars(total_loss=training_loss)
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # check if need to run eval step on validation data
            if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
                results = do_test(cfg, model)  # TODO:
                storage.put_scalars(**results['metrics'])

                if cfg.EARLY_STOPPING.ENABLE:
                    curr = None
                    if cfg.EARLY_STOPPING.MONITOR in results['metrics'].keys():
                        curr = results['metrics'][cfg.EARLY_STOPPING.MONITOR]

                    if curr is None:
                        logger.warning("Early stopping enabled but cannot find metric: %s" %
                                       cfg.EARLY_STOPPING.MONITOR)
                        logger.warning("Options for monitored metrics are: [%s]" %
                                       ", ".join(map(str, results['metrics'].keys())))
                    elif best_monitor_metric is None:
                        best_monitor_metric = curr
                    elif get_es_result(cfg.EARLY_STOPPING.MODE,
                                       curr, best_monitor_metric):
                        best_monitor_metric = curr
                        es_count = 0
                        logger.info("Best metric %s improved to %0.4f" %
                                    (cfg.EARLY_STOPPING.MONITOR, curr))
                        # update best model
                        periodic_checkpointer.save(name="model_best", **{**results['metrics']})
                        # save best metrics to a .txt file
                        with open(os.path.join(cfg.OUTPUT_DIR, 'best_metrics.txt'), 'w') as f:
                            json.dump(results['metrics'], f)
                    else:
                        logger.info("Early stopping metric %s did not improve, current %.04f, best %.04f" %
                                    (cfg.EARLY_STOPPING.MONITOR, curr, best_monitor_metric))
                        es_count += 1

                storage.put_scalar('val_loss', results['metrics']['val_loss'])

            # print out info about iteration
            # if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
            for writer in writers:
                writer.write()
            periodic_checkpointer.step(iteration)

            if es_count >= cfg.EARLY_STOPPING.PATIENCE:
                logger.info("Early stopping triggered, metric %s has not improved for %s validation steps" %
                            (cfg.EARLY_STOPPING.MONITOR, cfg.EARLY_STOPPING.PATIENCE))
                break


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True  # may result in a slowdown if set to True


def run(cfg):
    # setup logging
    setup_logger(output=cfg.OUTPUT_DIR)
    logger.info("Environment info:\n" + collect_env_info())

    path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with PathManager.open(path, "w") as f:
        f.write(cfg.dump())
    logger.info("Full config saved to {}".format(path))

    # make training deterministic
    seed_all(cfg.SEED)

    # get model and load onto device
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    # count number of parameters for model
    net_params = model.parameters()
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    logger.info("Number of model parameters: %.0f" % weight_count)

    return train(cfg, model)


if __name__ == '__main__':
    cfg = setup_config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    run(cfg)
