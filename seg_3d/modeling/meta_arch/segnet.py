import torch.nn as nn

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY


# TODO:
# - semantic segmentation head
# - preprocessing
@META_ARCH_REGISTRY.register()
class SemanticSegNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.backbone_name = cfg.MODEL.BACKBONE.NAME
        self.backbone = BACKBONE_REGISTRY.get(self.backbone_name)(**cfg.UNET)
        self.device = cfg.MODEL.DEVICE

    def forward(self, x):
        images = x.to(self.device)
        return self.backbone(images)
