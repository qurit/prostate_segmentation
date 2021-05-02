from functools import partial

from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

SOLVER_PARAMS = _C.SOLVER
# clear entire config
_C.clear()

new_CN = partial(CN, new_allowed=True)

# standard params
_C.MODEL = new_CN()
_C.MODEL.BACKBONE = new_CN()
_C.SOLVER = SOLVER_PARAMS
_C.SOLVER.PARAMS = new_CN()
_C.TEST = new_CN()

# eval only mode
_C.EVAL_ONLY = False

# early stopping mode
_C.EARLY_STOPPING = new_CN()

# loss
_C.LOSS = new_CN()
_C.LOSS.PARAMS = new_CN()

# unet
_C.UNET = new_CN()

# dataset
_C.DATASET = new_CN()

# transforms
_C.TRANSFORMS = new_CN()

_C.VERSION = 2
