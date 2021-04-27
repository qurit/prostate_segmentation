from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

SOLVER_PARAMS = _C.SOLVER
# clear entire config
_C.clear()

# standard params
_C.MODEL = CN()
_C.MODEL.BACKBONE = CN()
_C.SOLVER = SOLVER_PARAMS
_C.SOLVER.PARAMS = CN()
_C.TEST = CN()

# eval only mode
_C.EVAL_ONLY = False

# early stopping mode
_C.EARLY_STOPPING = CN()

# loss
_C.LOSS = CN()
_C.LOSS.PARAMS = CN()

# unet
_C.UNET = CN()

# dataset
_C.DATASET = CN()
