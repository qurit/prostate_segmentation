from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# data directory
_C.DATASET_PATH = 'data/image_dataset'

# eval only mode
_C.EVAL_ONLY = False

# early stopping mode
_C.EARLY_STOPPING = CN()
_C.EARLY_STOPPING.ENABLE = False
_C.EARLY_STOPPING.MONITOR = 'val_loss'
_C.EARLY_STOPPING.PATIENCE = 0
_C.EARLY_STOPPING.MODE = 'max'

# loss
_C.LOSS = CN()
_C.LOSS.PARAMS = CN()

# unet
_C.UNET = CN()
