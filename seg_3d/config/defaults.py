import torch
from fvcore.common.config import CfgNode as CN

_C = CN(new_allowed=True)

# CfgNodes can only contain a limited set of valid types
# _VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.BACKBONE = CN()
_C.MODEL.UNET = CN(new_allowed=True)

_C.MODEL.DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
_C.MODEL.WEIGHTS = ""  # specify path of a .pth file here containing model weights
_C.MODEL.META_ARCHITECTURE = "SemanticSegNet"
_C.MODEL.BACKBONE.NAME = "UNet3D"

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.PARAMS = CN(new_allowed=True)

_C.DATASET.FOLD = None  # kfold split index
_C.DATASET.DATA_SPLIT = "seg_3d/data/data_split_3fold.json"

_C.DATASET.TRAIN_DATASET_PATH = ("data/image_dataset",)
_C.DATASET.TEST_DATASET_PATH = ("data/image_dataset",)

_C.DATASET.TRAIN_NUM_PATIENTS = None
_C.DATASET.VAL_NUM_PATIENTS = None
_C.DATASET.TEST_NUM_PATIENTS = None

_C.DATASET.TRAIN_PATIENT_KEYS = None
_C.DATASET.VAL_PATIENT_KEYS = None
_C.DATASET.TEST_PATIENT_KEYS = None

_C.DATASET.PARAMS.modality_roi_map = [{"PT": ["Bladder"]}]
_C.DATASET.CLASS_LABELS = ["Background", "Bladder"]

# -----------------------------------------------------------------------------
# TRANSFORMS
# -----------------------------------------------------------------------------
_C.TRANSFORMS = CN(new_allowed=True)

# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.PARAMS = CN(new_allowed=True)

_C.LOSS.FN = "BCEDiceLoss"  # available loss functions are inside losses.py

# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
_C.SOLVER = CN(new_allowed=True)
_C.SOLVER.PARAMS = CN(new_allowed=True)

_C.SOLVER.OPTIM = "Adam"  # can select any optim from torch.optim
_C.SOLVER.PARAMS.lr = 0.0001
_C.SOLVER.IMS_PER_BATCH = 1
_C.SOLVER.MAX_ITER = 10000
_C.SOLVER.CHECKPOINT_PERIOD = 1000  # Save a checkpoint after every this number of iterations

# lr scheduler params
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (500,)  # The iteration number to decrease learning rate by GAMMA
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 0  # Number of iterations to increase lr to base lr

# -----------------------------------------------------------------------------
# EARLY STOPPING
# -----------------------------------------------------------------------------
_C.EARLY_STOPPING = CN()

_C.EARLY_STOPPING.PATIENCE = 0  # set to 0 to disable
_C.EARLY_STOPPING.MONITOR = "val_loss"
_C.EARLY_STOPPING.MODE = "min"  # options are 'max' and 'min' for either maximizing or minimizing early stopping metric

# -----------------------------------------------------------------------------
# EVAL
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.EVAL_PERIOD = 20  # The period (in terms of steps) to evaluate the model during training. Set to 0 to disable
# metrics which get computed during eval, available metrics found inside evaluation/metrics.py
# if none specified only val_loss is computed
_C.TEST.EVAL_METRICS = []
_C.TEST.FINAL_EVAL_METRICS = []  # metrics to compute for the final eval step at the end of training
_C.TEST.INFERENCE_FILE_NAME = None  # "inference.pk"  # name of file which stores results from evaluation, set to None to disable saving
_C.TEST.THRESHOLDS = None
# option to visualize predictions using the mask visualizer
# if in training mode, will only visualize masks at the end of the main train loop
_C.TEST.VIS_PREDS = False

# -----------------------------------------------------------------------------
# PIPELINE MODES
# -----------------------------------------------------------------------------
# Option to resume training, useful when training was interrupted. Loads iteration number, solver state, and model
# weights from .pth file specified in checkpoint file inside _C.OUTPUT_DIR
_C.RESUME = False
_C.EVAL_ONLY = False  # Option to only run evaluation on data specified by _C.DATASET.TEST_DATASET_PATH
_C.PRED_ONLY = False  # Option to only run inference on data specified by _C.DATASET.TEST_DATASET_PATH

# -----------------------------------------------------------------------------
# MISC
# -----------------------------------------------------------------------------
_C.NUM_WORKERS = 0  # number of workers for the data loaders, 0 means only the main process will load batches
_C.OUTPUT_DIR = None  # the output directory to store models, results, predictions, etc. for a training/evaluation run
_C.CONFIG_FILE = None  # file path to a config to load parameters from
_C.AMP_ENABLED = True  # enables automatic mixed precision training