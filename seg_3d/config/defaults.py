from fvcore.common.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.BACKBONE = CN()
_C.MODEL.UNET = CN(new_allowed=True)

_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = ""  # specify path of a .pth file here containing model weights
_C.MODEL.META_ARCHITECTURE = "SemanticSegNet"
_C.MODEL.BACKBONE.NAME = "UNet3D"

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.PARAMS = CN(new_allowed=True)

_C.DATASET.TRAIN_DATASET_PATH = "data/image_dataset"
_C.DATASET.TEST_DATASET_PATH = "data/test_dataset"

_C.DATASET.TRAIN_NUM_PATIENTS = None
_C.DATASET.VAL_NUM_PATIENTS = None

_C.DATASET.PARAMS.modality = "PT"
_C.DATASET.PARAMS.rois = ["Bladder"]
_C.DATASET.PARAMS.num_slices = 128  # number of slices in axial plane
_C.DATASET.PARAMS.crop_size = (128, 128)  # size of centre crop

_C.DATASET.CLASS_LABELS = ["Background", "Bladder"]

# -----------------------------------------------------------------------------
# TRANSFORMS
# -----------------------------------------------------------------------------
_C.TRANSFORMS = CN(new_allowed=True)

_C.TRANSFORMS.deform_sigma = None
_C.TRANSFORMS.crop = None
_C.TRANSFORMS.p_flip = None
_C.TRANSFORMS.div_by_max = False

# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.PARAMS = CN(new_allowed=True)

_C.LOSS.FN = "DiceLoss"  # available loss functions are inside losses.py

# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.PARAMS = CN(new_allowed=True)

_C.SOLVER.OPTIM = "Adam"  # can select any optim from torch.optim
_C.SOLVER.PARAMS.lr = 0.0001
_C.SOLVER.IMS_PER_BATCH = 1
_C.SOLVER.MAX_ITER = 10000
_C.SOLVER.CHECKPOINT_PERIOD = 1000  # Save a checkpoint after every this number of iterations

# lr scheduler params, see detectron2/solver/build.py for LR scheduler options
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (500,)  # The iteration number to decrease learning rate by GAMMA
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 0  # Number of iterations to increase lr to base lr

# -----------------------------------------------------------------------------
# EARLY STOPPING
# -----------------------------------------------------------------------------
_C.EARLY_STOPPING = CN()

_C.EARLY_STOPPING.PATIENCE = 0  # set to 0 to disable
_C.EARLY_STOPPING.MONITOR = "val_loss"
_C.EARLY_STOPPING.MODE = "min"

# -----------------------------------------------------------------------------
# EVAL
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.EVAL_PERIOD = 0  # The period (in terms of steps) to evaluate the model during training. Set to 0 to disable
# metrics which get computed during eval, available metrics found inside evaluation/metrics.py
# if none specified only val_loss is computed
_C.TEST.EVAL_METRICS = []

# -----------------------------------------------------------------------------
# PIPELINE MODES
# -----------------------------------------------------------------------------
# Option to resume training, useful when training was interrupted. Loads iteration number, solver state, and model
# weights from .pth file specified in checkpoint file inside _C.OUTPUT_DIR
_C.RESUME = False
_C.EVAL_ONLY = False  # Option to only run evaluation on data specified by _C.DATASET.TEST_DATASET_PATH

# -----------------------------------------------------------------------------
# MISC
# -----------------------------------------------------------------------------
_C.SEED = 99
_C.OUTPUT_DIR = "./output"
_C.CONFIG_FILE = None
