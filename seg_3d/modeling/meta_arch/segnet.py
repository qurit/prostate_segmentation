import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY


# TODO:
# - semantic segmentation head
# - preprocessing
@META_ARCH_REGISTRY.register()
class SemanticSegNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg, input_shape=ShapeSpec(channels=1))
        self.device = cfg.MODEL.DEVICE

    def forward(self, x):
        images = x.to(self.device)
        return self.backbone(images)
