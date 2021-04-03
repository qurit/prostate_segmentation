import os
import json
import pickle
import logging
import numpy as np
from time import time

from seg_3d.losses import get_loss_criterion
from seg_3d.evaluation.metrics import MetricList, get_metrics
from seg_3d.evaluation.evaluator import Evaluator
from seg_3d.data.dataset import ImageToImage3D
from seg_3d.config import get_cfg
from seg_3d.seg_utils import EarlyStopping, seed_all
import seg_3d.modeling.backbone.unet
import seg_3d.modeling.meta_arch.segnet

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
# - default directory
# - plot loss
# - use wandb and sacred
def setup_config():
    cfg = get_cfg()

    cfg.MODEL.DEVICE = "cpu"  # "cuda:0"
    cfg.SEED = 99

    # pipeline modes
    cfg.RESUME = False  # Option to resume training, useful when training was interrupted
    cfg.EVAL_ONLY = False
    cfg.TEST.EVAL_PERIOD = 20  # The period (in terms of steps) to evaluate the model during training. Set to 0 to disable
    cfg.TEST.EVAL_METRICS = ["dice_score"]  # metrics which get computed during eval
    #                        "classwise_iou",  # FIXME
    #                        "classwise_f1"]
    cfg.EARLY_STOPPING.PATIENCE = 10  # set to 0 to disable
    cfg.EARLY_STOPPING.MONITOR = "val_loss"
    cfg.EARLY_STOPPING.MODE = "max"

    # paths
    cfg.TRAIN_DATASET_PATH = "data/image_dataset"  # "/home/yous/Desktop/ryt/image_dataset"
    cfg.TEST_DATASET_PATH = "data/test_dataset"
    cfg.OUTPUT_DIR = "seg_3d/output/test-1"
    cfg.MODEL.WEIGHTS = ""  # file path for .pth model weight file, needs to be set when EVAL_ONLY or RESUME set to True

    # dataset options
    cfg.MODALITY = "PT"
    cfg.ROIS = ["Bladder"]
    # TODO: centre crop + other preprocessing options here

    # model architecture
    cfg.MODEL.META_ARCHITECTURE = "SemanticSegNet"
    cfg.MODEL.BACKBONE.NAME = "UNet3D"
    # specify UNet params which are defined in Abstract3DUNet
    cfg.UNET.in_channels = 1
    cfg.UNET.out_channels = 1
    cfg.UNET.f_maps = 8
    cfg.UNET.final_sigmoid = True  # final activation used during testing, if True then apply Sigmoid, else apply Softmax

    # loss
    cfg.LOSS.FN = "BCEDiceLoss"  # available loss functions are inside losses.py
    # specify loss params
    cfg.LOSS.PARAMS.alpha = 1.0
    cfg.LOSS.PARAMS.beta = 0.0

    # solver params
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 100  # Save a checkpoint after every this number of iterations
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (30000,)  # The iteration number to decrease learning rate by GAMMA
    cfg.SOLVER.WARMUP_ITERS = 0  # Number of iterations to increase lr to base lr

    return cfg


# TODO:
# - switch from default optim
# - could load up all scans into memory
def train(cfg, model):
    model.train()

    # get training dataset
    train_dataset = ImageToImage3D(dataset_path=cfg.TRAIN_DATASET_PATH, modality=cfg.MODALITY, rois=cfg.ROIS)
    val_dataset = ImageToImage3D(dataset_path=cfg.TEST_DATASET_PATH, modality=cfg.MODALITY, rois=cfg.ROIS)

    # get default optimizer (torch.optim.SGD) and scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # init loss criterion
    loss = get_loss_criterion(cfg)(**cfg.LOSS.PARAMS)
    logger.info("Loss:\n{}".format(loss))

    # init eval metrics and evaluator
    metric_list = MetricList(metrics=get_metrics(cfg))
    evaluator = Evaluator(device=cfg.MODEL.DEVICE, loss=loss, dataset=val_dataset, metric_list=metric_list)

    # init checkpointers
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    # TODO: test if continue from iteration when resume=True
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=cfg.RESUME).get("iteration", -1) + 1)
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    # init writers which periodically output/save metric scores
    writers = [CommonMetricPrinter(max_iter, window_size=1),
               JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
               TensorboardXWriter(cfg.OUTPUT_DIR)]

    # init early stopping
    early_stopping = EarlyStopping(monitor=cfg.EARLY_STOPPING.MONITOR,
                                   patience=cfg.EARLY_STOPPING.PATIENCE,
                                   mode=cfg.EARLY_STOPPING.MODE)
    early_stopping.check_is_valid(metric_list)

    # measuring the time elapsed
    train_start = time()
    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        # start main training loop
        for iteration, batched_inputs in zip(range(start_iter, max_iter),
                                             DataLoader(train_dataset, batch_size=1, shuffle=False)):

            storage.step()
            sample = batched_inputs["image"].unsqueeze(0).float()
            labels = batched_inputs["gt_mask"].float().to(cfg.MODEL.DEVICE)

            # do a forward pass, input is of shape (b, c, d, h, w)
            preds = model(sample).squeeze(0)

            optimizer.zero_grad()
            training_loss = loss(preds, labels)  # https://github.com/wolny/pytorch-3dunet#training-tips
            training_loss.backward()
            optimizer.step()

            storage.put_scalars(training_loss=training_loss, lr=optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # check if need to run eval step on validation data
            if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
                results = evaluator.evaluate(model)
                storage.put_scalars(**results["metrics"])

                # check early stopping
                if early_stopping.check_early_stopping(results["metrics"]):
                    # update best model
                    periodic_checkpointer.save(name="model_best", **results["metrics"])
                    # save inference results
                    with open(os.path.join(cfg.OUTPUT_DIR, 'inference.pk'), 'wb') as f:
                        pickle.dump(results["inference"], f, protocol=pickle.HIGHEST_PROTOCOL)
                    # save best metrics to a .txt file
                    with open(os.path.join(cfg.OUTPUT_DIR, 'best_metrics.txt'), 'w') as f:
                        json.dump(results["metrics"], f)

                elif early_stopping.triggered:
                    # do something before finishing execution?
                    break

            # print out info about iteration
            # if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
            for writer in writers:
                writer.write()
            periodic_checkpointer.step(iteration)
            # TODO: print training time


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
    weight_count = sum(np.prod(param.size()) for param in net_params)
    logger.info("Number of model parameters: %.0f" % weight_count)

    if cfg.EVAL_ONLY:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        return NotImplementedError

    return train(cfg, model)


if __name__ == '__main__':
    cfg = setup_config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    run(cfg)
