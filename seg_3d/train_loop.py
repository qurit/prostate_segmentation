import glob
import json
import logging
import os
import pickle
import random
from time import time

import numpy as np
from fvcore.common.config import CfgNode as CN
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from iopath import PathManager
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import MongoObserver
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import seg_3d
from seg_3d.config import get_cfg
from seg_3d.data.dataset import ImageToImage3D, JointTransform3D, Image3D
from seg_3d.evaluation.evaluator import Evaluator
from seg_3d.evaluation.metrics import MetricList, get_metrics
from seg_3d.losses import get_loss_criterion, get_optimizer
import seg_3d.modeling.backbone.unet
import seg_3d.modeling.meta_arch.segnet
from seg_3d.modeling.meta_arch.segnet import build_model
from seg_3d.utils.early_stopping import EarlyStopping
from seg_3d.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, EventStorage
from seg_3d.utils.logger import setup_logger, add_fh
from seg_3d.utils.scheduler import build_lr_scheduler
from seg_3d.utils.misc_utils import seed_all, TrainingSampler
from seg_3d.utils.tb_formatter import DefaultTensorboardFormatter

SETTINGS.CONFIG.READ_ONLY_CONFIG = False  # allows us to update config based on run name (hacky)
ex = Experiment()  # initialize Sacred experiment https://sacred.readthedocs.io/

# global vars
ROOT_OUTPUT_DIR = "seg_3d/output"            # root directory for all runs
ROOT_MASK_DIR = "masks/"                     # root directory to store figures for prediction masks
MODEL_BEST_FILE_NAME = "model_best.pth"      # best model file
CONFIG_FILE_NAME = "config.yaml"             # stores all the parameters for a run
BEST_METRICS_FILE_NAME = "best_metrics.txt"  # stores the metrics of the best model
METRICS_WRITER_FILE_NAME = "metrics.json"    # stores metrics computed during each training/eval step


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

    # if no patient keys specified for val then pass in the patients keys from excluded set in train (hacky)
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
    metric_list = MetricList(metrics=get_metrics(cfg.TEST.EVAL_METRICS), class_labels=cfg.LOSS.PARAMS.class_labels)
    evaluator = Evaluator(device=cfg.MODEL.DEVICE, loss=loss, dataset=val_dataset, num_workers=cfg.NUM_WORKERS,
                          metric_list=metric_list, amp_enabled=cfg.AMP_ENABLED)

    # init checkpointers
    checkpointer = Checkpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=cfg.RESUME).get("iteration", -1) + 1)
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    # init writers which periodically output/save metric scores
    # window_size gives option to do median smoothing of metrics
    writers = [CommonMetricPrinter(max_iter, window_size=1),
               JSONWriter(os.path.join(cfg.OUTPUT_DIR, METRICS_WRITER_FILE_NAME), window_size=1),
               TensorboardXWriter(cfg.OUTPUT_DIR, window_size=1)]

    # init tensorboard formatter for images
    # tensorboard is used to visualize predictions during training steps
    # https://www.tensorflow.org/tensorboard
    tensorboard_img_formatter = DefaultTensorboardFormatter()

    # init early stopping
    early_stopping = EarlyStopping(monitor=cfg.EARLY_STOPPING.MONITOR,
                                   patience=cfg.EARLY_STOPPING.PATIENCE,
                                   mode=cfg.EARLY_STOPPING.MODE)
    early_stopping.check_is_valid(list(metric_list.metrics.keys()), cfg.DATASET.CLASS_LABELS)

    # setup for automatic mixed precision (AMP) training
    scaler = GradScaler(enabled=cfg.AMP_ENABLED)

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
                        num_workers=cfg.NUM_WORKERS, worker_init_fn=random.seed(cfg.seed),
                        sampler=TrainingSampler(size=len(train_dataset), shuffle=True, seed=cfg.seed))
            ):

                storage.step()

                # extract data from item recieved from data loader
                patients = batched_inputs["patient"]
                logger.info(('Patients ' + '{}, ' * len(patients)).format(*patients))
                orig_imgs = batched_inputs["orig_image"]
                sample = batched_inputs["image"].to(cfg.MODEL.DEVICE)
                labels = batched_inputs["gt_mask"].to(cfg.MODEL.DEVICE)
                data = {'labels': labels,
                        'dist_map': batched_inputs["dist_map"].to(cfg.MODEL.DEVICE)}  # dist map is an input to boundary loss

                optimizer.zero_grad()

                # runs the forward pass with autocasting if enabled
                with autocast(enabled=cfg.AMP_ENABLED):
                    # do a forward pass, input is of shape (N, C, D, H, W)
                    preds = model(sample)
                    training_loss = loss(preds, data)  # https://github.com/wolny/pytorch-3dunet#training-tips

                    # check if need to process masks and images to be visualized in tensorboard
                    for idx, p in enumerate(patients):
                        # hardcoded, only visualize 4 patients from train set
                        if p in train_dataset.patient_keys[:4]:
                            for name, batch in zip(["img_orig", "img_aug", "mask_gt", "mask_pred"],
                                                   [orig_imgs, sample, labels, preds]):
                                tags_imgs = tensorboard_img_formatter(name=p + "/" + name,
                                                                      batch=batch[idx].unsqueeze(0).detach().cpu())

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

                scalars = {"training_loss": training_loss,
                           "lr": optimizer.param_groups[0]["lr"],
                           **loss_dict}
                for k, v in scalars.items():
                    # a bit redundant, but scalars are stored in 2 places: Sacred and EventStorage
                    storage.put_scalar(k, v, smoothing_hint=False)
                    ex.log_scalar(k, float(v), step=iteration)

                scheduler.step()

                # check if need to run eval step on validation data
                if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
                    results = evaluator.evaluate(model)
                    for k, v in results["metrics"].items():
                        storage.put_scalar(k, v, smoothing_hint=False)
                        ex.log_scalar(k, float(v), step=iteration)

                    # check early stopping
                    if early_stopping.check_early_stopping(results["metrics"]):
                        # update best model
                        periodic_checkpointer.save(name="model_best", iteration=iteration, **results["metrics"])
                        # save inference results
                        if cfg.TEST.INFERENCE_FILE_NAME:
                            with open(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.INFERENCE_FILE_NAME), "wb") as f:
                                pickle.dump(results["inference"], f, protocol=pickle.HIGHEST_PROTOCOL)
                        # save best metrics to a .txt file
                        with open(os.path.join(cfg.OUTPUT_DIR, BEST_METRICS_FILE_NAME), "w") as f:
                            json.dump(results["metrics"], f, indent=4)

                    elif early_stopping.triggered:
                        break

                # print out info about iteration
                for writer in writers:
                    writer.write()
                # images have been written to tensorboard file so clear them from memory
                storage.clear_images()
                periodic_checkpointer.step(iteration)

        finally:
            # add more logic here to do something before finishing execution
            train_time = time() - train_start
            logger.info("Completed training in %.0f s (%.2f h)" % (train_time, train_time / 3600))

            # run final evaluation with best model
            model_checkpoint = os.path.join(cfg.OUTPUT_DIR, MODEL_BEST_FILE_NAME)

            if cfg.TEST.FINAL_EVAL_METRICS and model_checkpoint in checkpointer.get_all_checkpoint_files():
                logger.info("Running final evaluation with best model...")
                evaluator.thresholds = cfg.TEST.THRESHOLDS

                # add weight file to db
                ex.add_artifact(os.path.join(cfg.OUTPUT_DIR, MODEL_BEST_FILE_NAME), content_type="weights")
                # load best model
                checkpointer.load(model_checkpoint, checkpointables=["model"])

                # add new metrics to metric list
                metric_list.metrics = get_metrics(cfg.TEST.FINAL_EVAL_METRICS)

                # configure mask visualizer if specified
                if cfg.TEST.VIS_PREDS:
                    evaluator.set_mask_visualizer(
                        cfg.DATASET.CLASS_LABELS[1:], os.path.join(cfg.OUTPUT_DIR, ROOT_MASK_DIR)  # skip label for bgd
                    )

                # run evaluation
                results = evaluator.evaluate(model)
                for k, v in results["metrics"].items():
                    ex.log_scalar(k, float(v), step=iteration)

                # save inference results (can be a large file)
                if cfg.TEST.INFERENCE_FILE_NAME:
                    with open(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.INFERENCE_FILE_NAME), "wb") as f:
                        pickle.dump(results["inference"], f, protocol=pickle.HIGHEST_PROTOCOL)
                # save best metrics to a .txt file
                with open(os.path.join(cfg.OUTPUT_DIR, BEST_METRICS_FILE_NAME), "w") as f:
                    json.dump(results["metrics"], f, indent=4)


@ex.main
def main(_config, _run):
    if "LOAD_ONLY_CFG_FILE" in _config and _config["LOAD_ONLY_CFG_FILE"]:
        cfg.merge_from_other_cfg(CN(_config))
        # next two lines are for when config file is specified in cmdline
        cfg.merge_from_file(CN(_config).CONFIG_FILE)  # any param inside config file will be overwritten here
        cfg.OUTPUT_DIR = None

    else:
        cfg.merge_from_other_cfg(CN(_config))  # this merges the param changes done in cmd line

    name = _run.experiment_info["name"]
    base_dir = os.path.join(ROOT_OUTPUT_DIR, name)

    # make training deterministic
    seed_all(cfg.seed)

    if cfg.DATASET.FOLD is not None:
        base_dir = os.path.join(base_dir, str(cfg.DATASET.FOLD))
        data_split = json.load(open(cfg.DATASET.DATA_SPLIT, "r"))
        if cfg.DATASET.TRAIN_PATIENT_KEYS is None:
            cfg.DATASET.TRAIN_PATIENT_KEYS = data_split[str(cfg.DATASET.FOLD)]["train"]["keys"]
        if cfg.DATASET.VAL_PATIENT_KEYS is None:
            cfg.DATASET.VAL_PATIENT_KEYS = data_split[str(cfg.DATASET.FOLD)]["val"]["keys"]
        if cfg.DATASET.TEST_PATIENT_KEYS is None:
            cfg.DATASET.TEST_PATIENT_KEYS = data_split[str(cfg.DATASET.FOLD)]["test"]["keys"]

    if any([cfg.EVAL_ONLY, cfg.PRED_ONLY, cfg.RESUME]) and not cfg.MODEL.WEIGHTS:
        # get model weight file if not specified
        cfg.MODEL.WEIGHTS = os.path.join(base_dir, MODEL_BEST_FILE_NAME)
        print(os.path.join(base_dir, MODEL_BEST_FILE_NAME))
        assert os.path.isfile(cfg.MODEL.WEIGHTS)

    if cfg.OUTPUT_DIR is None:
        if cfg.EVAL_ONLY:
            # create a new directory for this eval run
            prefix = str(len(glob.glob(os.path.join(base_dir, "eval*"))))
            cfg.OUTPUT_DIR = os.path.join(base_dir, "eval_" + prefix)
        elif cfg.PRED_ONLY:
            # create a new directory for this pred run
            prefix = str(len(glob.glob(os.path.join(base_dir, "pred*"))))
            cfg.OUTPUT_DIR = os.path.join(base_dir, "pred_" + prefix)
        else:
            cfg.OUTPUT_DIR = base_dir

    cfg.freeze()  # freeze all parameters i.e. no more changes can be made to config
    # make sure latest version of config is saved to mongo db (hacky)
    if ex.observers != 0 and ex.observers[0].run_entry is not None:
        ex.observers[0].run_entry["config"] = cfg

    # save logs to output directory
    for log in logger_list:
        add_fh(log, output=cfg.OUTPUT_DIR)
    logger.info("Starting new run...")

    # create directory to store output files
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg_path = os.path.join(cfg.OUTPUT_DIR, CONFIG_FILE_NAME)

    # check if file already exists
    if os.path.isfile(cfg_path):
        logger.warning("Config file {} already exists! Renaming old one...".format(cfg_path))
        os.rename(cfg_path, cfg_path + ".bk")

    with PathManager().open(cfg_path, "w") as f:
        f.write(cfg.dump())
    logger.info("Full config saved to {}".format(cfg_path))
    _run.add_artifact(cfg_path, 'config')  # make its easier to download from omniboard

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
                                      num_patients=cfg.DATASET.TEST_NUM_PATIENTS,
                                      patient_keys=cfg.DATASET.TEST_PATIENT_KEYS,
                                      class_labels=cfg.DATASET.CLASS_LABELS,
                                      joint_transform=eval_transforms,
                                      **cfg.DATASET.PARAMS)

        # init eval metrics and evaluator
        metric_list = MetricList(metrics=get_metrics(cfg.TEST.FINAL_EVAL_METRICS), class_labels=cfg.DATASET.CLASS_LABELS)
        evaluator = Evaluator(device=cfg.MODEL.DEVICE, dataset=test_dataset,
                              metric_list=metric_list, thresholds=cfg.TEST.THRESHOLDS)

        # configure mask visualizer if specified
        if cfg.TEST.VIS_PREDS:
            evaluator.set_mask_visualizer(
                cfg.DATASET.CLASS_LABELS[1:], os.path.join(cfg.OUTPUT_DIR, ROOT_MASK_DIR)
            )

        results = evaluator.evaluate(model)
        for k, v in results["metrics"].items():
            # add to sacred experiment
            ex.log_scalar(k, float(v), step=0)
        # save inference results
        if cfg.TEST.INFERENCE_FILE_NAME:
            with open(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.INFERENCE_FILE_NAME), "wb") as f:
                pickle.dump(results["inference"], f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    elif cfg.PRED_ONLY:  # TODO
        logger.info("Running inference only!")

        return NotImplementedError

        # load model
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS, checkpointables=["model"])

        # get dataset for inference
        test_dataset = Image3D(dataset_path=cfg.DATASET.TEST_DATASET_PATH)

        # get predictions

        # save results

    return train(model)


@ex.config
def config():
    # pipeline params

    # ###########################################
    # option to load config file here
    # cfg.CONFIG_FILE = "seg_3d/config/prostate-config.yaml"
    # cfg.merge_from_file(cfg.CONFIG_FILE)

    ## can add more config changes here ##

    # ###########################################

    # by default, set the output directory name based on the specified experiment name via `--name` or `-n` in cmd line
    cfg.OUTPUT_DIR = None

    # add to sacred experiment
    ex.add_config(cfg)  # NOTE: for run_configs.sh script everything above in config() should be commented out!!

    # sacred params
    seed = 99  # comment this out to disable deterministic experiments
    # next two params are useful for sorting through runs with Sacred
    tags = [i for i in cfg.DATASET.CLASS_LABELS if i != "Background"]  # add ROIs as tags
    tags.extend([list(i.keys())[0] for i in cfg.DATASET.PARAMS.modality_roi_map])  # add modalities as tags


if __name__ == '__main__':
    cfg = get_cfg()  # config global variable
    logger_list = [
        setup_logger(name="fvcore"),  # fvcore is a light-weight core library from facebook
        setup_logger(name=seg_3d.__name__)
    ]
    logger = logging.getLogger(seg_3d.__name__ + "." + __name__)

    # mongo observer - the recommendeded way of storing run information from Sacred
    # to ignore oberservers use flag -u, useful for some quick tests or debugging runs
    ex.observers.append(
        MongoObserver(url=f'mongodb://'
                          'sample:password'  # credentials are set in .env file
                          # f'{os.environ["MONGO_INITDB_ROOT_USERNAME"]}:'
                          # f'{os.environ["MONGO_INITDB_ROOT_PASSWORD"]}'
                          f'@localhost:27017/?authMechanism=SCRAM-SHA-1', db_name='db')
    )  # assumes mongo db is running
    ex.logger = logger
    ex.run_commandline()
