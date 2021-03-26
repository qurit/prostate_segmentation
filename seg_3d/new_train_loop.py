import os
import logging
import numpy as np

from seg_3d.data.dataset import ImageToImage3D
from seg_3d.config import get_cfg
import seg_3d.modeling.backbone.unet
import seg_3d.modeling.meta_arch.segnet

from torch.utils.data import DataLoader
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, EventStorage
from detectron2.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger("detectron2")


# TODO:
# - resume option
# - loads weights
# - default directory
# - implementation for eval
# - plot loss
# - distributed training
# - seed all
# - use wandb and sacred
# - early stopping
def setup_config():
    cfg = get_cfg()

    cfg.MODEL.DEVICE = "cpu"

    # pipeline modes
    cfg.EVAL_ONLY = False
    cfg.TEST.EVAL_PERIOD = 0  # The period (in terms of steps) to evaluate the model during training. Set to 0 to disable
    cfg.EARLY_STOPPING.ENABLE = False
    cfg.EARLY_STOPPING.PATIENCE = 10

    # paths
    cfg.DATASET_PATH = "data/image_dataset"
    cfg.OUTPUT_DIR = "seg_3d/output/test-1"

    # dataset options
    cfg.MODALITY = "CT"
    cfg.ROIS = ["Bladder"]  # FIXME: some patients with no prostate?

    # model architecture
    cfg.MODEL.META_ARCHITECTURE = "SemanticSegNet"
    cfg.MODEL.BACKBONE.NAME = "build_unet3d_backbone"

    # solver params
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 10  # Save a checkpoint after every this number of iterations
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (30000,)  # The iteration number to decrease learning rate by GAMMA

    return cfg


# TODO:
# - switch from default optim
# - evaluate on val set
# - could load up all scans into memory
def train(cfg, model):
    model.train()

    # get training dataset
    dataset = ImageToImage3D(dataset_path=cfg.DATASET_PATH, modality=cfg.MODALITY, rois=cfg.ROIS)

    # get default optimizer (torch.optim.SGD) and scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # init checkpointers
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    # init writers which periodically output/save metric scores
    writers = [CommonMetricPrinter(max_iter),
               JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
               TensorboardXWriter(cfg.OUTPUT_DIR)]

    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        # start main training loop
        for iteration, batched_inputs in zip(range(start_iter, max_iter),
                                             DataLoader(dataset, batch_size=1, shuffle=False)):
            storage.step()

            # do a forward pass
            # preds = model(batched_inputs["image"])  # FIXME

            # optimizer.zero_grad()
            # training_loss = self.loss(y_out, y_batch)
            # training_loss.backward()
            # optimizer.step()

            # storage.put_scalars(total_loss=losses_reduced)
            # storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            # scheduler.step()

            # print out info about iteration
            for writer in writers:
                writer.write()
            periodic_checkpointer.step(iteration)


def run(cfg):
    # get model and load onto device
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    # count number of parameters for model
    net_params = model.parameters()
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    logger.info("Number of model parameters: %.0f" % weight_count)

    return train(cfg, model)


if __name__ == '__main__':
    cfg = setup_config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    run(cfg)