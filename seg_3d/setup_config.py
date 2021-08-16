import os

from fvcore.common.config import CfgNode as CN

from seg_3d.config import get_cfg
import seg_3d.modeling.backbone.unet
import seg_3d.modeling.meta_arch.segnet
from seg_3d.utils.cross_validation import k_folds


def setup_config(*args) -> CN:
    # simple setup
    if args:
        cfg = get_cfg()
        cfg.merge_from_file(args[0][0])
        yield cfg
        return

    base_path = "seg_3d/output/"  # convenient variable if all paths have same base path
    config_paths = []  # specify a list of file paths for existing configs

    # specify params to change for each run to launch consecutive trainings
    # each inner list corresponds to the list of keys, values to change for a particular run
    # e.g. param_search = [["A", 1, "B", 2], ["C", 3"]] -> in 1st run set param A to 1 and param B to 2, in 2nd run set param C to 3
    # NOTE: training runs will be overwritten if OUTPUT_DIR is not unique
    param_search = []

    # option to run cross validation
    k_fold = None  # k_folds(n_splits=59, subjects=59)

    # iterate over either config paths, param search, or folds
    for i, args in enumerate(config_paths or param_search or k_fold):
        # get the default config from default.py
        cfg = get_cfg()

        if type(args) is list:
            # loads params from args
            cfg.merge_from_list(args)

        elif type(args) is str:
            # load params from existing yaml
            cfg.CONFIG_FILE = os.path.join(base_path, args)
            cfg.merge_from_file(cfg.CONFIG_FILE)

        elif k_fold:
            # load params from existing yaml
            cfg.CONFIG_FILE = os.path.join(base_path, "config.yaml")  # edit here
            cfg.merge_from_file(cfg.CONFIG_FILE)

            # get indices for current fold
            train_idx, test_idx = args
            cfg.DATASET.TRAIN_PATIENT_KEYS = train_idx.astype(int).tolist()
            cfg.DATASET.VAL_PATIENT_KEYS = test_idx.astype(int).tolist()
            cfg.OUTPUT_DIR = os.path.join(base_path, "cross-val-run", str(i))  # edit here

        # option to resume training
        # cfg.RESUME = True

        # add custom config which override parameter values if they already exist
        # add_custom_config(cfg)
        # add_inference_config(cfg, weights=os.path.join(base_path, "model_best.pth"))

        yield cfg


def add_inference_config(cfg: CN, weights="model_best.pth") -> None:
    cfg.EVAL_ONLY = True
    cfg.MODEL.WEIGHTS = weights
    cfg.TEST.INFERENCE_FILE_NAME = "test_inference.pk"
    cfg.MODEL.UNET.final_sigmoid = False
    cfg.TEST.THRESHOLDS = None


def add_custom_config(cfg: CN) -> None:
    pass
    # cfg.OUTPUT_DIR = "test-1"

    # dataset and transform
    # cfg.TRANSFORMS

    # evaluation
    # cfg.TEST

    # loss
    # cfg.LOSS

    # optimizer and lr scheduler
    # cfg.SOLVER
