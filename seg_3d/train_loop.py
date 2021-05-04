import json
import logging
import os
import pickle
from time import time

import numpy as np
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data.samplers import TrainingSampler
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, EventStorage
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from torch.utils.data import DataLoader

import seg_3d
from seg_3d.data.dataset import ImageToImage3D, JointTransform2D
from seg_3d.evaluation.evaluator import Evaluator
from seg_3d.evaluation.metrics import MetricList, get_metrics
from seg_3d.losses import get_loss_criterion, get_optimizer
from seg_3d.seg_utils import EarlyStopping, seed_all, DefaultTensorboardFormatter
from seg_3d.setup_config import setup_config


def train(cfg, model):
    model.train()

    # get training and validation datasets
    transform_augmentations = JointTransform2D(test=False, **cfg.TRANSFORMS)
    train_dataset = ImageToImage3D(joint_transform=transform_augmentations, dataset_path=cfg.TRAIN_DATASET_PATH,
                                   num_patients=cfg.TRAIN_NUM_PATIENTS, **cfg.DATASET)
    val_transforms = JointTransform2D(test=True, **cfg.TRANSFORMS)
    val_dataset = ImageToImage3D(joint_transform=val_transforms, dataset_path=cfg.TRAIN_DATASET_PATH, num_patients=cfg.VAL_NUM_PATIENTS,
                                 patient_keys=train_dataset.excluded_patients, **cfg.DATASET)
    logger.info("Patient keys excluded from train-val split: {}".format(val_dataset.excluded_patients))

    # setup logger for detectron2 modules
    setup_logger(output=cfg.OUTPUT_DIR, name="detectron2")

    # get optimizer specified in config file
    optimizer = get_optimizer(cfg)(model.parameters(), **cfg.SOLVER.PARAMS)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # init loss criterion
    loss = get_loss_criterion(cfg)(**cfg.LOSS.PARAMS)
    logger.info("Loss:\n{}".format(loss))

    # init eval metrics and evaluator
    metric_list = MetricList(metrics=get_metrics(cfg), class_labels=cfg.class_labels)
    evaluator = Evaluator(device=cfg.MODEL.DEVICE, loss=loss, dataset=val_dataset, metric_list=metric_list)

    # init checkpointers
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    checkpointer.logger = logging.getLogger("detectron2.checkpoint")
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=cfg.RESUME).get("iteration", -1) + 1)
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    # init writers which periodically output/save metric scores
    writers = [CommonMetricPrinter(max_iter, window_size=1),
               JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
               TensorboardXWriter(cfg.OUTPUT_DIR)]

    # init tensorboard formatter for images
    tensorboard_img_formatter = DefaultTensorboardFormatter()

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
        for iteration, batched_inputs in zip(
                range(start_iter, max_iter),
                DataLoader(train_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                           sampler=TrainingSampler(size=len(train_dataset), shuffle=True, seed=cfg.SEED))
        ):

            storage.step()
            sample = batched_inputs["image"]
            labels = batched_inputs["gt_mask"].squeeze(1).long().to(cfg.MODEL.DEVICE)

            # do a forward pass, input is of shape (N, C, D, H, W)
            preds = model(sample).squeeze(1)

            optimizer.zero_grad()
            training_loss = loss(preds, labels)  # https://github.com/wolny/pytorch-3dunet#training-tips
            training_loss.backward()
            optimizer.step()

            storage.put_scalars(training_loss=training_loss, lr=optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # process masks and images to be visualized in tensorboard
            for name, batch in zip(["img_orig", "img_aug", "mask_gt", "mask_pred"],
                                   [batched_inputs["orig_image"], sample, labels, preds]):
                tags_imgs = tensorboard_img_formatter(name=name, batch=batch.detach().cpu())

                # add each tag image tuple to tensorboard
                for item in tags_imgs:
                    storage.put_image(*item)

            # check if need to run eval step on validation data
            if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
                results = evaluator.evaluate(model)
                storage.put_scalars(**results["metrics"])

                # check early stopping
                if early_stopping.check_early_stopping(results["metrics"]):
                    # update best model
                    periodic_checkpointer.save(name="model_best", iteration=iteration, **results["metrics"])
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

    train_time = time() - train_start
    logger.info("Completed training in %.0f s (%.2f h)" % (train_time, train_time / 3600))


def run(cfg):
    # logger.info("Environment info:\n" + collect_env_info())

    # check if need to load config from disk
    if cfg.CONFIG_FILE:
        logger.warning("Merging config with {}. All params specified inside this file will overwrite existing values!"
                       .format(cfg.CONFIG_FILE))
        cfg.merge_from_file(cfg.CONFIG_FILE)

    path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with PathManager.open(path, "w") as f:
        f.write(cfg.dump())
    logger.info("Full config saved to {}".format(path))
    cfg.freeze()  # freezes all param values

    # make training deterministic
    seed_all(cfg.SEED)

    # get model and load onto device
    model = build_model(cfg)
    # model = smp.UnetPlusPlus(
    #         encoder_name='densenet161',
    #         in_channels=1,
    #         classes=3
    #     )

    # logger.info("Model:\n{}".format(model))

    # count number of parameters for model
    net_params = model.parameters()
    weight_count = sum(np.prod(param.size()) for param in net_params)
    logger.info("Number of model parameters: %.0f" % weight_count)

    if cfg.EVAL_ONLY:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        return NotImplementedError

    return train(cfg, model)


if __name__ == '__main__':
    # setup config
    cfg = setup_config()

    # setup logging
    setup_logger(output=cfg.OUTPUT_DIR, name=seg_3d.__name__)
    logger = logging.getLogger(seg_3d.__name__ + "." + __name__)

    # create directory to store output files
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # run train loop
    run(cfg)
