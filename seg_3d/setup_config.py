from fvcore.common.config import CfgNode as CN

from seg_3d.config import get_cfg
import seg_3d.modeling.backbone.unet
import seg_3d.modeling.meta_arch.segnet


def setup_config() -> CN:
    # get the default config from default.py
    cfg = get_cfg()

    # load params from existing yaml
    cfg.CONFIG_FILE = "seg_3d/config/bladder-detection.yaml"
    cfg.merge_from_file(cfg.CONFIG_FILE)

    # option to resume training
    # resume_training(cfg)

    # add custom config which override parameter values if they already exist
    add_custom_config(cfg)
    # add_inference_config(cfg)

    return cfg


def add_custom_config(cfg: CN) -> None:
    cfg.OUTPUT_DIR = "seg_3d/output/test-1"

    # dataset and transform
    cfg.DATASET.TRAIN_NUM_PATIENTS = 40
    cfg.DATASET.VAL_NUM_PATIENTS = 19
    cfg.TRANSFORMS.deform_sigma = 5

    # evaluation
    cfg.TEST.EVAL_METRICS = ["classwise_dice_score", "argmax_dice_score", "overlap"]
    cfg.EARLY_STOPPING.PATIENCE = 200  # set to 0 to disable
    cfg.EARLY_STOPPING.MONITOR = "classwise_dice_score/Bladder"
    cfg.EARLY_STOPPING.MODE = "max"

    # loss
    cfg.LOSS.FN = "CEDiceLoss"
    cfg.LOSS.PARAMS.ce_weight = 0.
    cfg.LOSS.PARAMS.dice_weight = 1.0
    cfg.LOSS.PARAMS.overlap_weight = 10.
    cfg.LOSS.PARAMS.class_weight = [1, 3, 1]
    cfg.LOSS.PARAMS.device = cfg.MODEL.DEVICE

    # optimizer and lr scheduler
    cfg.SOLVER.PARAMS.lr = 0.0001
    cfg.SOLVER.IMS_PER_BATCH = 5
    cfg.SOLVER.MAX_ITER = 1000000
    cfg.SOLVER.CHECKPOINT_PERIOD = 200
    cfg.SOLVER.STEPS = (240, 480, 700,)


def add_inference_config(cfg: CN) -> None:
    cfg.EVAL_ONLY = True
    cfg.MODEL.WEIGHTS = "seg_3d/output/test-1/model_best.pth"
    cfg.TEST.INFERENCE_FILE_NAME = "test_inference.pk"


def resume_training(cfg: CN) -> None:
    cfg.RESUME = True
