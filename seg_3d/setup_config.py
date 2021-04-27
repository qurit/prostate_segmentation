import os
from datetime import datetime

from seg_3d.config import get_cfg
import seg_3d.modeling.backbone.unet
import seg_3d.modeling.meta_arch.segnet


def setup_config():
    cfg = get_cfg()

    cfg.MODEL.DEVICE = "cpu"  # "cuda:0"
    cfg.SEED = 99

    # pipeline modes
    cfg.RESUME = False  # Option to resume training, useful when training was interrupted
    cfg.EVAL_ONLY = False
    cfg.TEST.EVAL_PERIOD = 20  # The period (in terms of steps) to evaluate the model during training. Set to 0 to disable
    cfg.TEST.EVAL_METRICS = ["dice_score", "iou", "f1"]  # metrics which get computed during eval
    cfg.EARLY_STOPPING.PATIENCE = 10  # set to 0 to disable
    cfg.EARLY_STOPPING.MONITOR = "dice_score"
    cfg.EARLY_STOPPING.MODE = "max"

    # paths
    cfg.TRAIN_DATASET_PATH = "data/image_dataset"
    cfg.TEST_DATASET_PATH = "data/test_dataset"
    cfg.OUTPUT_DIR = "seg_3d/output/test-3"
    cfg.MODEL.WEIGHTS = ""  # file path for .pth model weight file, needs to be set when EVAL_ONLY or RESUME set to True

    # dataset options
    cfg.TRAIN_NUM_PATIENTS = 40
    cfg.VAL_NUM_PATIENTS = 15
    cfg.DATASET.modality = "PT"
    cfg.DATASET.rois = ["Bladder"]
    cfg.DATASET.num_slices = 128  # number of slices in axial plane
    cfg.DATASET.crop_size = (128, 128)  # size of centre crop
    cfg.DATASET.one_hot_mask = False  # False or int for num of classes

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
    # specify loss params (if any)
    cfg.LOSS.PARAMS.bce_weight = 0.0
    cfg.LOSS.PARAMS.dice_weight = 1.0

    # optim
    cfg.SOLVER.OPTIM = "Adam"  # can select any optim from torch.optim
    cfg.SOLVER.PARAMS.lr = 0.001
    cfg.SOLVER.PARAMS.weight_decay = 0
    # cfg.SOLVER.PARAMS.momentum = 0.9
    cfg.SOLVER.IMS_PER_BATCH = 3
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 100  # Save a checkpoint after every this number of iterations

    # lr scheduler params
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (30000,)  # The iteration number to decrease learning rate by GAMMA
    cfg.SOLVER.WARMUP_ITERS = 0  # Number of iterations to increase lr to base lr

    # make a default dir
    if not cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = os.path.join("seg_3d/output", str(datetime.now().strftime('%m-%d_%H:%M/')))

    return cfg
