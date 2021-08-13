import json
import logging
import os
import pickle
import random
from time import time

import numpy as np
import torch
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from iopath import PathManager
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import seg_3d
from seg_3d.data.dataset import ImageToImage3D, JointTransform3D, Image3D
from seg_3d.evaluation.evaluator import Evaluator
from seg_3d.evaluation.metrics import MetricList, get_metrics
from seg_3d.losses import get_loss_criterion, get_optimizer
from seg_3d.modeling.meta_arch.segnet import build_model
from seg_3d.setup_config import setup_config
from seg_3d.utils.early_stopping import EarlyStopping
from seg_3d.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, EventStorage
from seg_3d.utils.logger import setup_logger
from seg_3d.utils.scheduler import build_lr_scheduler
from seg_3d.utils.misc_utils import seed_all, TrainingSampler, plot_loss, zip_files_in_dir
from seg_3d.utils.tb_formatter import DefaultTensorboardFormatter


def train(model):
    model.train()

    # get training and validation datasets
    train_transforms = JointTransform3D(test=False, **cfg.TRANSFORMS)
    train_dataset = ImageToImage3D(joint_transform=train_transforms,
                                   dataset_path=cfg.DATASET.TRAIN_DATASET_PATH,
                                   num_patients=cfg.DATASET.TRAIN_NUM_PATIENTS,
                                   patient_keys=cfg.DATASET.TRAIN_PATIENT_KEYS,
                                   class_labels=cfg.DATASET.CLASS_LABELS,
                                   **cfg.DATASET.PARAMS)

    # if no patient keys specified for val then pass in the patients keys from excluded set in train
    if cfg.DATASET.VAL_PATIENT_KEYS is None:
        cfg.DATASET.defrost()
        cfg.DATASET.VAL_PATIENT_KEYS = train_dataset.excluded_patients
        cfg.freeze()

    val_transforms = JointTransform3D(test=True, **cfg.TRANSFORMS)
    val_dataset = ImageToImage3D(joint_transform=val_transforms,
                                 dataset_path=cfg.DATASET.TRAIN_DATASET_PATH,
                                 num_patients=cfg.DATASET.VAL_NUM_PATIENTS,
                                 patient_keys=cfg.DATASET.VAL_PATIENT_KEYS,
                                 class_labels=cfg.DATASET.CLASS_LABELS,
                                 **cfg.DATASET.PARAMS)

    assert len(np.intersect1d(train_dataset.patient_keys, val_dataset.patient_keys)) == 0,\
        "duplicate patients in train and val split!"

    # get optimizer specified in config file
    optimizer = get_optimizer(cfg.SOLVER.OPTIM)(model.parameters(), **cfg.SOLVER.PARAMS)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # init loss criterion
    loss = get_loss_criterion(cfg.LOSS.FN)(**cfg.LOSS.PARAMS)
    logger.info("Loss:\n{}".format(loss))

    # init eval metrics and evaluator
    metric_list = MetricList(metrics=get_metrics(cfg.TEST.EVAL_METRICS), class_labels=cfg.DATASET.CLASS_LABELS)
    evaluator = Evaluator(device=cfg.MODEL.DEVICE, loss=loss, dataset=val_dataset, num_workers=cfg.NUM_WORKERS,
                          metric_list=metric_list, amp_enabled=cfg.AMP_ENABLED, **cfg.DATASET.PARAMS)

    # init checkpointers
    checkpointer = Checkpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=cfg.RESUME).get("iteration", -1) + 1)
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    # init writers which periodically output/save metric scores
    # window_size gives option to do median smoothing of metrics
    writers = [CommonMetricPrinter(max_iter, window_size=1),
               JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), window_size=1),
               TensorboardXWriter(cfg.OUTPUT_DIR, window_size=1)]

    # init tensorboard formatter for images
    tensorboard_img_formatter = DefaultTensorboardFormatter()

    # init early stopping
    early_stopping = EarlyStopping(monitor=cfg.EARLY_STOPPING.MONITOR,
                                   patience=cfg.EARLY_STOPPING.PATIENCE,
                                   mode=cfg.EARLY_STOPPING.MODE)
    early_stopping.check_is_valid(list(metric_list.metrics.keys()), cfg.DATASET.CLASS_LABELS)

    # measuring the time elapsed
    train_start = time()
    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        try:
            # start main training loop
            for iteration, batched_inputs in zip(
                    range(start_iter, max_iter),
                    DataLoader(
                        train_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                        num_workers=cfg.NUM_WORKERS, worker_init_fn=random.seed(cfg.SEED),
                        sampler=TrainingSampler(size=len(train_dataset), shuffle=True, seed=cfg.SEED))
            ):

                storage.iter = iteration
                sample = batched_inputs["image"]
                labels = batched_inputs["gt_mask"].to(cfg.MODEL.DEVICE)

                optimizer.zero_grad()

                # runs the forward pass with autocasting if enabled
                with autocast(enabled=cfg.AMP_ENABLED):
                    # do a forward pass, input is of shape (N, C, D, H, W)
                    preds = model(sample)
                    training_loss = loss(preds, labels)  # https://github.com/wolny/pytorch-3dunet#training-tips

                    # check if need to process masks and images to be visualized in tensorboard
                    if iteration - start_iter < 5 or (iteration + 1) % 40 == 0:
                        for name, batch in zip(["img_orig", "img_aug", "mask_gt", "mask_pred"],
                                               [batched_inputs["orig_image"], sample, labels, preds]):
                            tags_imgs = tensorboard_img_formatter(name=name, batch=batch.detach().cpu())

                            # add each tag image tuple to tensorboard
                            for item in tags_imgs:
                                storage.put_image(*item)

                # loss can either return a dict of losses or just a single tensor
                loss_dict = {}
                if type(training_loss) is dict:
                    loss_dict = {"loss/" + k: v.item() for k, v in training_loss.items()}
                    training_loss = sum(training_loss.values())

                # scales the loss, and calls backward() to create scaled gradients
                scaler.scale(training_loss).backward()

                # unscales gradients and calls or skips optimizer.step()
                scaler.step(optimizer)

                # updates the scale for next iteration
                scaler.update()

                storage.put_scalars(training_loss=training_loss,
                                    lr=optimizer.param_groups[0]["lr"],
                                    **loss_dict, smoothing_hint=False)
                scheduler.step()

                # check if need to run eval step on validation data
                if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
                    results = evaluator.evaluate(model)
                    storage.put_scalars(**results["metrics"], smoothing_hint=False)

                    # check early stopping
                    if early_stopping.check_early_stopping(results["metrics"]):
                        # update best model
                        periodic_checkpointer.save(name="model_best", iteration=iteration, **results["metrics"])
                        # save inference results
                        with open(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.INFERENCE_FILE_NAME), "wb") as f:
                            pickle.dump(results["inference"], f, protocol=pickle.HIGHEST_PROTOCOL)
                        # save best metrics to a .txt file
                        with open(os.path.join(cfg.OUTPUT_DIR, "best_metrics.txt"), "w") as f:
                            json.dump(results["metrics"], f, indent=4)

                    elif early_stopping.triggered:
                        break

                # print out info about iteration
                # if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()
                # images have been written to tensorboard file so clear them from memory
                storage.clear_images()
                periodic_checkpointer.step(iteration)

        finally:
            # add more logic here to do something before finishing execution
            train_time = time() - train_start
            logger.info("Completed training in %.0f s (%.2f h)" % (train_time, train_time / 3600))

            # plot loss curve
            path = os.path.join(cfg.OUTPUT_DIR, "model_loss.png")
            try:
                plot_loss(path, storage)
                logger.info("Saved model loss figure at {}".format(path))
            except KeyError:
                logger.info("Not enough metric information to plot loss, skipping...")

            # run final evaluation with best model
            model_checkpoint = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
            if cfg.TEST.FINAL_EVAL_METRICS and model_checkpoint in checkpointer.get_all_checkpoint_files():
                logger.info("Running final evaluation with best model...")
                # load best model
                checkpointer.load(model_checkpoint, checkpointables=["model"])

                # add new metrics to metric list
                metric_list.metrics = get_metrics(cfg.TEST.FINAL_EVAL_METRICS)

                # configure mask visualizer if specified
                if cfg.TEST.VIS_PREDS:
                    evaluator.set_mask_visualizer(
                        cfg.DATASET.CLASS_LABELS[1:], os.path.join(cfg.OUTPUT_DIR, "masks")  # skip label for bgd
                    )

                # run evaluation and save metrics to a .txt file
                with open(os.path.join(cfg.OUTPUT_DIR, "best_metrics.txt"), "w") as f:
                    json.dump(
                        evaluator.evaluate(model)["metrics"], f, indent=4
                    )


def run():
    path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with PathManager().open(path, "w") as f:
        f.write(cfg.dump())
    logger.info("Full config saved to {}".format(path))

    # save zipped up code to output dir
    zip_files_in_dir(seg_3d.__name__, zip_file_name=os.path.join(cfg.OUTPUT_DIR, "code.zip"))

    # make training deterministic
    seed_all(cfg.SEED)

    # get model and load onto device
    model = build_model(cfg)

    # count number of parameters for model
    net_params = model.parameters()
    weight_count = sum(np.prod(param.size()) for param in net_params)
    logger.info("Number of model parameters: %.0f" % weight_count)

    if cfg.EVAL_ONLY:
        logger.info("Running evaluation only!")
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS, checkpointables=["model"])

        eval_transforms = JointTransform3D(test=True, **cfg.TRANSFORMS)
        # get dataset for evaluation
        test_dataset = ImageToImage3D(dataset_path=cfg.DATASET.TEST_DATASET_PATH,
                                      patient_keys=cfg.DATASET.TEST_PATIENT_KEYS,
                                      class_labels=cfg.DATASET.CLASS_LABELS,
                                      joint_transform=eval_transforms,
                                      **cfg.DATASET.PARAMS)

        # init eval metrics and evaluator
        metric_list = MetricList(metrics=get_metrics(cfg.TEST.EVAL_METRICS), class_labels=cfg.DATASET.CLASS_LABELS)
        evaluator = Evaluator(device=cfg.MODEL.DEVICE, dataset=test_dataset,
                              metric_list=metric_list, thresholds=cfg.TEST.THRESHOLDS)

        # configure mask visualizer if specified
        if cfg.TEST.VIS_PREDS:
            evaluator.set_mask_visualizer(
                cfg.DATASET.CLASS_LABELS[1:], os.path.join(cfg.OUTPUT_DIR, "masks")
            )

        results = evaluator.evaluate(model)
        # save inference results
        with open(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.INFERENCE_FILE_NAME), "wb") as f:
            pickle.dump(results["inference"], f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    elif cfg.PRED_ONLY:  # TODO
        logger.info("Running inference only!")
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS, checkpointables=["model"])

        # get dataset for inference
        test_dataset = Image3D(dataset_path=cfg.DATASET.TEST_DATASET_PATH)

        # get predictions

        # save results

        return NotImplementedError

    return train(model)


if __name__ == '__main__':
    cfg_gen = setup_config()  # this returns a generator

    for cfg in cfg_gen:
        cfg.MODEL.DEVICE = "cpu" if not torch.cuda.is_available() else cfg.MODEL.DEVICE
        cfg.freeze()

        # setup for automatic mixed precision (AMP) training
        scaler = GradScaler(enabled=cfg.AMP_ENABLED)

        # setup loggers for the various modules
        logger_list = [
            setup_logger(output=cfg.OUTPUT_DIR, name="fvcore"),
            setup_logger(output=cfg.OUTPUT_DIR, name=seg_3d.__name__)
        ]

        logger = logging.getLogger(seg_3d.__name__ + "." + __name__)
        logger.info("Starting new run...")

        # create directory to store output files
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # run train loop
        run()

        # cleanup run
        for log in logger_list:
            log.handlers.clear()

