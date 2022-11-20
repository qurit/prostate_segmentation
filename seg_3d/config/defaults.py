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

_C.DATASET.TRAIN_DATASET_PATH = ("data/image_dataset",)
_C.DATASET.TEST_DATASET_PATH = ("data/test_dataset",)

_C.DATASET.TRAIN_NUM_PATIENTS = None
_C.DATASET.VAL_NUM_PATIENTS = None
_C.DATASET.TEST_NUM_PATIENTS = None

_C.DATASET.TRAIN_PATIENT_KEYS = None
_C.DATASET.VAL_PATIENT_KEYS = None
_C.DATASET.TEST_PATIENT_KEYS = None

_C.DATASET.PARAMS.modality_roi_map = [{"CT": ["Bladder"]}]
_C.DATASET.PARAMS.num_slices = None  # number of slices in axial plane, if None then selects shortest scan length from dataset
_C.DATASET.PARAMS.crop_size = None  # size of centre crop, if None then no centre cropping done

_C.DATASET.PARAMS.patch_size = None  # 3 dim tuple for patch size
_C.DATASET.PARAMS.patch_stride = None  # 3 dim tuple for patch stride (how far to move between patches)
_C.DATASET.PARAMS.patch_halo = None  # 3 dim tuple for size of halo to be removed from patches
_C.DATASET.PARAMS.patching_input_size = None  # Tuple describing original image input size before patching
_C.DATASET.PARAMS.patching_label_size = None  # Tuple describing original labels/mask size before patching
_C.DATASET.PARAMS.attend_samples = False
_C.DATASET.PARAMS.attend_samples_all_axes = False
_C.DATASET.PARAMS.mask_samples = False
_C.DATASET.PARAMS.drop_ct = False
_C.DATASET.PARAMS.attend_frame_dict_path = ("seg_3d/data/attend_frame_range.npy")

_C.DATASET.CLASS_LABELS = ["Background", "Bladder"]

# -----------------------------------------------------------------------------
# TRANSFORMS
# -----------------------------------------------------------------------------
_C.TRANSFORMS = CN(new_allowed=True)

_C.TRANSFORMS.deform_sigma = None
_C.TRANSFORMS.crop = None
_C.TRANSFORMS.p_flip = None
_C.TRANSFORMS.div_by_max = False
_C.TRANSFORMS.multi_scale = None
_C.TRANSFORMS.ignore_bg = False

# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.PARAMS = CN(new_allowed=True)

_C.LOSS.FN = "DiceLoss"  # available loss functions are inside losses.py

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
_C.EARLY_STOPPING.MODE = "min"

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
_C.NUM_WORKERS = 6  # number of workers for the data loaders
_C.OUTPUT_DIR = None
_C.CONFIG_FILE = None
_C.AMP_ENABLED = True  # enables automatic mixed precision training