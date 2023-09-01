# original code from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from fvcore.common.registry import Registry
from torch import nn as nn, einsum


LOSS_REGISTRY = Registry('LOSS')


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.weight = weight
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # return Dice scores for all channels/classes
        return 1. - per_channel_dice


@LOSS_REGISTRY.register()
class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight=None):
        return compute_per_channel_dice(input, target, weight=weight)


@LOSS_REGISTRY.register()
class SurfaceLoss(nn.Module):
    """
    Original code from
    https://github.com/LIVIAETS/boundary-loss/blob/8f4457416a583e33cae71443779591173e27ec62/losses.py#L76

    Computes the Boundary Loss as described in https://arxiv.org/pdf/1812.07032.pdf
    """
    def __init__(self, class_weight=None, idc=None):
        super(SurfaceLoss, self).__init__()
        
        self.idc = idc
        self.class_weight = class_weight
        
        if self.idc is not None:
            assert len(self.idc) == len(self.class_weight)

    def forward(self, logits: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:

        pc = nn.Softmax(dim=1)(logits).type(torch.float32)
        dc = dist_maps.type(torch.float32)

        if self.idc is not None:
            pc = pc[:, self.idc, ...]
            dc = dc[:, self.idc, ...]

        if self.class_weight is not None:
            multipled = einsum("bkxyz,bkxyz->bkxyz", pc, dc)
            multipled = multipled.mean(dim=(2,3,4)).squeeze()
            multipled *= self.class_weight.to(pc.device)

        else:

            multipled = einsum("bkxyz,bkxyz->bkxyz", pc, dc)

        loss = multipled.mean()

        return loss


@LOSS_REGISTRY.register()
class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf"""

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


@LOSS_REGISTRY.register()
class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, bce_weight, dice_weight, normalization="softmax", class_labels=None, class_weight=None, class_balanced=False, gdl=False):
        """
        Args:
            bce_weight: multiplies BCE loss by 'bce_weight'
            dice_weight: multiplies dice loss by 'dice_weight'
            normalization: normalization of model outputs, either 'softmax' or 'sigmoid'
            class_labels: class labels to make it easier to parse the channels of the prediction mask
            class_weight: weights assigned to each class in the losses
            class_balanced: weights assigned to each class in the losses that account for class imbalance
            gdl: whether to use GDL (generalized dice loss) or regular soft dice loss
        """
        super(BCEDiceLoss, self).__init__()
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.class_weight = None if not class_weight else torch.as_tensor(class_weight, dtype=torch.float)
        
        self.dice_weight = dice_weight
        self.normalization = normalization
        self.gdl = gdl
        self.dice = DiceLoss(normalization=normalization) if not self.gdl else GeneralizedDiceLoss(normalization=normalization)

        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        if self.class_weight is not None:
            self.bce.pos_weight = self.class_weight.view(1, -1, 1, 1, 1).to(self.device)
        
        self.class_labels = class_labels
        self.logger = logging.getLogger(__name__)

        self.class_balanced = class_balanced
    
    @staticmethod
    def get_class_balanced_weights(target):
        epsilon = 1e-10
        num_classes = target.size()[1]
        weights = [target[:, 0, ...].sum() / (target[:, x, ...].sum() + epsilon) for x in range(num_classes)]
        return torch.as_tensor(weights, dtype=torch.float)

    def forward(self, input, data):
        target = data['labels']
    
        if self.class_balanced:     
            self.class_weight = self.get_class_balanced_weights(target)
            self.bce.pos_weight = self.class_weight.view(1, -1, 1, 1, 1).to(self.device)

        dice_loss = self.dice(input, target)
        dice_verbose = 1 - dice_loss.detach().cpu().numpy()
        dice_loss *= self.class_weight.to(self.device) if not self.gdl else 1 # apply per channel weighting to dice

        # print the individual dice scores
        if dice_loss.shape:
            if self.class_labels is not None:
                dice_labels_tuple = [i for i in zip(self.class_labels, dice_verbose)]
                class_weight_labels_tuple = [i for i in zip(self.class_labels, self.class_weight)]
                dice_log = ["{} - {:.4f}, ".format(*i) for i in dice_labels_tuple]
                class_weight_log = ["{} - {:.4f}, ".format(*i) for i in class_weight_labels_tuple]
            else:
                dice_log = ["{:.4f}, ".format(i) for i in dice_verbose]
                class_weight_log = ["{} - {:.4f}, ".format(*i) for i in self.class_weight]

            self.logger.info(("Dice: " + "{}" * len(dice_log)).format(*dice_log))

            if self.class_balanced:
                self.logger.info(("Class weights: " + "{}" * len(class_weight_log)).format(*class_weight_log))
        
        return {
            "dice": self.dice_weight * dice_loss.sum(),
            "bce": self.bce_weight * self.bce(input, target)
        }


@LOSS_REGISTRY.register()
class BCEDiceOverlapLoss(BCEDiceLoss):
    """Linear combination of BCE, Dice, and overlap losses"""

    def __init__(self, bce_weight, dice_weight, overlap_weight, overlap_idx=(1, 2), normalization="softmax",
                 class_labels=None, class_weight=None, class_balanced=False, gdl=False):
        """
        Augments the BCEDiceLoss loss with an additional loss which penalizes overlap between predicted bladder
        (index overlap_idx[0]) and ground truth tumor (index overlap_idx[1]).
        """
        super().__init__(bce_weight, dice_weight, normalization=normalization, class_labels=class_labels, class_weight=class_weight, class_balanced=class_balanced, gdl=gdl)

        self.overlap_weight = overlap_weight
        self.overlap_idx = overlap_idx  # tuple containing the channel indices of pred, gt for overlap computation

    @staticmethod
    def overlap(pred_chan, gt_chan, pred, gt) -> torch.Tensor:
        """Penalizes voxels that are predicted bladder but are labeled ground truth tumor"""
        pred = nn.Softmax(dim=1)(pred)
        # get the right channels from pred and gt tensors
        pred = pred[:, pred_chan, ...]
        gt = gt[:, gt_chan, ...]

        if gt.sum() != 0:
            return (pred * gt).sum() / gt.sum()

        return torch.tensor(0.)

    def forward(self, input, data) -> Dict[str, torch.Tensor]:
        target = data['labels']
        if input.shape[1] < target.shape[1]:
            loss_dict = super().forward(input, {'labels': target[:, :input.shape[1], ...]})
        else:
            loss_dict = super().forward(input, data)

        # don't compute overlap if overlap_idx is set to None
        overlap_loss = self.overlap(*self.overlap_idx, input, target) if self.overlap_idx else torch.tensor(0.)

        loss_dict['overlap'] = self.overlap_weight * overlap_loss

        return loss_dict


@LOSS_REGISTRY.register()
class BoundaryBCEDiceLoss(nn.Module):
    """Linear combination of BCE, Dice, and Boundary loss"""

    def __init__(self, bce_weight=1, dice_weight=1, surface_weight=1, alpha_weight=None, iter_per_alpha_update=None,
                 normalization="softmax", class_labels=None, class_weight=None, class_balanced=False, gdl=False):
        """
        Args:
            bce_weight: multiplies BCE loss by 'bce_weight'
            dice_weight: multiplies dice loss by 'dice_weight'
            surface_weight: multiplies boundary loss by 'surface_weight'
            alpha_weight: initial value for the rebalancing of weights strategy between the weighting 
                of dice and boundary described in https://arxiv.org/pdf/1812.07032.pdf 
            iter_per_alpha_update: number of iterations between updates of 'alpha_weight'
            normalization: normalization of model outputs, either 'softmax' or 'sigmoid'
            class_labels: class labels to make it easier to parse the channels of the prediction mask
            class_weight: weights assigned to each class in the losses
            class_balanced: weights assigned to each class in the losses that account for class imbalance
            gdl: whether to use GDL (generalized dice loss) or regular soft dice loss
        """
        super(BoundaryBCEDiceLoss, self).__init__()
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.class_weight = None if not class_weight else torch.as_tensor(class_weight, dtype=torch.float)

        self.alpha_weight = alpha_weight
        assert 0 <= self.alpha_weight <= 1
        self.iter_per_alpha_update = iter_per_alpha_update
        if alpha_weight or iter_per_alpha_update: assert alpha_weight is not None and iter_per_alpha_update is not None

        self.surface = SurfaceLoss(self.class_weight)
        self.surface_weight = surface_weight
        
        self.dice_weight = dice_weight
        self.batch_count = 0
        self.normalization = normalization
        self.gdl = gdl
        self.dice = DiceLoss(normalization=normalization) if not self.gdl else GeneralizedDiceLoss(normalization=normalization)

        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        if self.class_weight is not None:
            self.bce.pos_weight = self.class_weight.view(1, -1, 1, 1, 1).to(self.device)
        
        self.class_labels = class_labels
        self.logger = logging.getLogger(__name__)

        self.class_balanced = class_balanced
        
    @staticmethod
    def get_class_balanced_weights(target):
        epsilon = 1e-10
        num_classes = target.size()[1]
        weights = [target[:, 0, ...].sum() / (target[:, x, ...].sum() + epsilon) for x in range(num_classes)]
        return torch.as_tensor(weights, dtype=torch.float)

    def update_alpha_weight(self):
        if self.alpha_weight is None:
            return
        if self.batch_count > 0 and (self.batch_count % self.iter_per_alpha_update == 0):
            if self.alpha_weight >= 0.02:
                self.alpha_weight -= 0.01
            else:
                self.alpha_weight = 0.01

        self.batch_count += 1
        self.logger.info(("alpha weight: {}").format(self.alpha_weight))

    def forward(self, input, data):
        self.update_alpha_weight()

        target = data['labels']
        distms = data['dist_map']
    
        if self.class_balanced:     
            self.class_weight = self.get_class_balanced_weights(target)
            self.bce.pos_weight = self.class_weight.view(1, -1, 1, 1, 1).to(self.device)

        dice_loss = self.dice(input, target)
        dice_verbose = 1 - dice_loss.detach().cpu().numpy()
        dice_loss *= self.class_weight.to(self.device) if not self.gdl else 1 # apply per channel weighting to dice

        # print the individual dice scores
        if dice_loss.shape:
            if self.class_labels is not None:
                dice_labels_tuple = [i for i in zip(self.class_labels, dice_verbose)]
                dice_log = ["{} - {:.4f}, ".format(*i) for i in dice_labels_tuple]
            else:
                dice_log = ["{:.4f}, ".format(i) for i in dice_verbose]

            self.logger.info(("Dice: " + "{}" * len(dice_log)).format(*dice_log))

        if self.alpha_weight is not None:
            return {
                "dice": self.alpha_weight * dice_loss.sum(),
                "bce": self.bce_weight * self.bce(input, target),
                "boundary": (1 - self.alpha_weight) * self.surface(input, distms)
            }

        return {
            "dice": self.dice_weight * dice_loss.sum(),
            "bce": self.bce_weight * self.bce(input, target),
            "boundary": self.surface_weight * self.surface(input, distms)
        }


# HELPERS #
def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    assert input.size() == target.size()

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)

    if weight is not None:
        intersect = weight.to(input.device) * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)

    # smoothing term added to both numerator and denominator https://arxiv.org/pdf/2207.09521.pdf
    dice = (2 * intersect + epsilon) / (denominator + epsilon)

    return dice


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(loss: str):
    return LOSS_REGISTRY.get(loss)


def get_optimizer(optim: str):
    return OPTIM_REGISTRY.get(optim)


# register all losses from torch.nn to LOSS registry
dir_loss = dir(torch.nn)
losses = [item for item in dir_loss if "Loss" in item]
for loss in losses:
    LOSS_REGISTRY.register(eval("torch.nn." + loss))


# register all optim from torch.optim to a registry
OPTIM_REGISTRY = Registry('OPTIM')
dir_optim = dir(torch.optim)
optims = [item for item in dir_optim if item[0].isupper()]
for o in optims:
    OPTIM_REGISTRY.register(eval("torch.optim." + o))
