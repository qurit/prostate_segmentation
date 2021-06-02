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


def train(model):
    model.train()

    # get training and validation datasets
    train_transforms = JointTransform2D(test=False, **cfg.TRANSFORMS)
    train_dataset = ImageToImage3D(joint_transform=train_transforms,
                                   dataset_path=cfg.DATASET.TRAIN_DATASET_PATH,
                                   num_patients=cfg.DATASET.TRAIN_NUM_PATIENTS,
                                   patient_keys=cfg.DATASET.TRAIN_PATIENT_KEYS,
                                   **cfg.DATASET.PARAMS)

    # if no patient keys specified for val then pass in the patients keys from excluded set in train
    if cfg.DATASET.VAL_PATIENT_KEYS is None:
        cfg.DATASET.defrost()
        cfg.DATASET.VAL_PATIENT_KEYS = train_dataset.excluded_patients
        cfg.freeze()

    val_transforms = JointTransform2D(test=True, **cfg.TRANSFORMS)
    val_dataset = ImageToImage3D(joint_transform=val_transforms,
                                 dataset_path=cfg.DATASET.TRAIN_DATASET_PATH,
                                 num_patients=cfg.DATASET.VAL_NUM_PATIENTS,
                                 patient_keys=cfg.DATASET.VAL_PATIENT_KEYS,
                                 **cfg.DATASET.PARAMS)

    assert len(np.intersect1d(train_dataset.patient_keys, val_dataset.patient_keys)) == 0,\
        "duplicate patients in train and val split!"

    # get optimizer specified in config file
    optimizer = get_optimizer(cfg)(model.parameters(), **cfg.SOLVER.PARAMS)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # init loss criterion
    loss = get_loss_criterion(cfg)(**cfg.LOSS.PARAMS)
    logger.info("Loss:\n{}".format(loss))

    # init eval metrics and evaluator
    metric_list = MetricList(metrics=get_metrics(cfg), class_labels=cfg.DATASET.CLASS_LABELS)
    evaluator = Evaluator(device=cfg.MODEL.DEVICE, loss=loss, dataset=val_dataset, metric_list=metric_list)

    # init checkpointers
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
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
    early_stopping.check_is_valid(metric_list, cfg.DATASET.CLASS_LABELS)

    # measuring the time elapsed
    train_start = time()
    logger.info("Starting training from iteration {}".format(start_iter))

    try:
        with EventStorage(start_iter) as storage:
            # start main training loop
            for iteration, batched_inputs in zip(
                    range(start_iter, max_iter),
                    DataLoader(train_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                               sampler=TrainingSampler(size=len(train_dataset), shuffle=True, seed=cfg.SEED))
            ):

                storage.step()
                sample = batched_inputs["image"]
                labels = batched_inputs["gt_mask"].to(cfg.MODEL.DEVICE)

                # do a forward pass, input is of shape (N, C, D, H, W)
                preds = model(sample)

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
                periodic_checkpointer.step(iteration)

    finally:
        # add more logic here to do something before finishing execution
        train_time = time() - train_start
        logger.info("Completed training in %.0f s (%.2f h)" % (train_time, train_time / 3600))


def run():
    path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with PathManager.open(path, "w") as f:
        f.write(cfg.dump())
    logger.info("Full config saved to {}".format(path))

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
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)

        # get dataset for evaluation
        test_dataset = ImageToImage3D(dataset_path=cfg.DATASET.TEST_DATASET_PATH,
                                      patient_keys=cfg.DATASET.TEST_PATIENT_KEYS,
                                      **cfg.DATASET.PARAMS)

        # init eval metrics and evaluator
        metric_list = MetricList(metrics=get_metrics(cfg), class_labels=cfg.DATASET.CLASS_LABELS)
        evaluator = Evaluator(device=cfg.MODEL.DEVICE, dataset=test_dataset,
                              metric_list=metric_list, thresholds=cfg.TEST.THRESHOLDS)

        results = evaluator.evaluate(model)
        # save inference results
        with open(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.INFERENCE_FILE_NAME), "wb") as f:
            pickle.dump(results["inference"], f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    return train(model)


if __name__ == '__main__':
    # specify params to change for each run to launch consecutive trainings
    # each inner list corresponds to the list of keys, values to change for a particular run
    # e.g. param_search = [["A", 1, "B", 2], ["C", 3"]] -> in 1st run set param A to 1 and param B to 2, in 2nd run set param C to 3
    # NOTE: training runs will be overwritten if OUTPUT_DIR is not unique
    param_search = [[]]  # empty list will run a single training

    for params in param_search:
        cfg = setup_config(*params)
        cfg.freeze()

        # setup loggers for the various modules
        setup_logger(output=cfg.OUTPUT_DIR, name="detectron2")
        setup_logger(output=cfg.OUTPUT_DIR, name="fvcore")
        setup_logger(output=cfg.OUTPUT_DIR, name=seg_3d.__name__)
        logger = logging.getLogger(seg_3d.__name__ + "." + __name__)

        # create directory to store output files
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # run train loop
        run()
