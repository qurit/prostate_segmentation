from fvcore.common.config import CfgNode as CN

from seg_3d.config import get_cfg
import seg_3d.modeling.backbone.unet
import seg_3d.modeling.meta_arch.segnet


def setup_config(*args) -> CN:
    # get the default config from default.py
    cfg = get_cfg()

    # loads params from args
    # cfg.merge_from_list(list(args))

    # load params from existing yaml
    cfg.CONFIG_FILE = args[0]
    cfg.merge_from_file(cfg.CONFIG_FILE)

    # option to resume training
    # resume_training(cfg)

    # add custom config which override parameter values if they already exist
    # add_custom_config(cfg)
    # add_inference_config(cfg)

    return cfg


def add_custom_config(cfg: CN) -> None:
    cfg.OUTPUT_DIR = "seg_3d/output/test-1"

    # dataset and transform
    cfg.TRANSFORMS.deform_sigma = 5
    cfg.TRANSFORMS.deform_points = (2, 2, 2)
    cfg.TRANSFORMS.crop = (128, 128)
    cfg.TRANSFORMS.p_flip = 0.5

    # evaluation
    cfg.TEST.EVAL_METRICS = ["classwise_dice_score", "argmax_dice_score", "overlap"]
    cfg.EARLY_STOPPING.PATIENCE = 40  # set to 0 to disable
    cfg.EARLY_STOPPING.MONITOR = "classwise_dice_score/Bladder"
    cfg.EARLY_STOPPING.MODE = "max"

    # loss
    cfg.LOSS.FN = "BCEDiceWithOverlapLoss"
    cfg.LOSS.PARAMS.bce_weight = 0.
    cfg.LOSS.PARAMS.dice_weight = 1.0
    cfg.LOSS.PARAMS.overlap_weight = 10.
    # tuple containing the channel indices of pred, gt for overlap computation: (pred bladder channel, gt tumor channel)
    cfg.LOSS.PARAMS.overlap_idx = (1, 2)
    cfg.LOSS.PARAMS.class_weight = [0, 2, 1]  # background, bladder, tumor weight

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
    cfg.MODEL.UNET.final_sigmoid = False
    cfg.TEST.THRESHOLDS = None


def resume_training(cfg: CN) -> None:
    cfg.RESUME = True
