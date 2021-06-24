import torch
import torch.nn as nn
from fvcore.common.registry import Registry

from seg_3d.modeling.backbone.unet import BACKBONE_REGISTRY

META_ARCH_REGISTRY = Registry('META_ARCH')


@META_ARCH_REGISTRY.register()
class SemanticSegNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.backbone_name = cfg.MODEL.BACKBONE.NAME
        self.backbone = BACKBONE_REGISTRY.get(self.backbone_name)(**cfg.MODEL.UNET)
        self.device = cfg.MODEL.DEVICE

    @property
    def final_activation(self):
        return self.backbone.final_activation

    def forward(self, x):
        images = x.to(self.device)
        return self.backbone(images)


def build_model(cfg):
    model = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)(cfg)
    return model.to(torch.device(cfg.MODEL.DEVICE))
