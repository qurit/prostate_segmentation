# original code from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
import logging
from math import exp
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from fvcore.common.registry import Registry
from torch import nn as nn, einsum
from torch.autograd import Variable

from seg_3d.utils.misc_utils import expand_as_one_hot

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
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

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

    def __init__(self, bce_weight, dice_weight, normalization="softmax", class_labels=None, class_weight=None, class_balanced=False):
        super(BCEDiceLoss, self).__init__()
        
        self.class_weight = None if not class_weight else torch.as_tensor(class_weight, dtype=torch.float)
        
        self.dice_weight = dice_weight
        self.normalization = normalization
        self.dice = DiceLoss(normalization=normalization, weight=self.class_weight)

        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.bce_class_weight = self.class_weight.view(1, len(class_weight), 1, 1, 1).to(self.device)
        
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.bce_class_weight)
        self.bce_weight = bce_weight
        
        self.class_labels = class_labels
        self.logger = logging.getLogger(__name__)

        self.class_balanced = class_balanced

    @staticmethod
    def get_class_balanced_weights(target):
        epsilon = 1e-10
        num_classes = target.size()[1]
        weights = [target[:, 0, ...].sum() / (target[:, x, ...].sum() + epsilon) for x in range(num_classes)]
        return torch.as_tensor(weights, dtype=torch.float)

    def forward(self, input, target):
        
        target = target['labels']

        dice_loss = self.dice(input, target)
        dice_verbose = 1 - dice_loss.detach().cpu().numpy()

        if self.class_labels is not None:
            dice_labels_tuple = [i for i in zip(self.class_labels, dice_verbose)]
            dice_log = ["{} - {:.4f}, ".format(*i) for i in dice_labels_tuple]
        else:
            dice_log = ["{:.4f}, ".format(i) for i in dice_verbose]

        self.logger.info(("Dice: " + "{}" * len(dice_log)).format(*dice_log))

        if self.class_balanced:
            weights = self.get_class_balanced_weights(target)
            self.dice = DiceLoss(normalization=self.normalization, weight=weights)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=weights.view(1, -1, 1, 1, 1).to(self.device))
        
        return {
            "dice": self.dice_weight * dice_loss.sum(),
            "bce": self.bce_weight * self.bce(input, target)
        }


@LOSS_REGISTRY.register()
class BCEDiceOverlapLoss(BCEDiceLoss):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, bce_weight, dice_weight, overlap_weight, overlap_idx=(1, 2), normalization="softmax",
         class_labels=None, class_weight=None, class_balanced=False):
        
        super().__init__(bce_weight, dice_weight, normalization=normalization, class_labels=class_labels, class_weight=class_weight, class_balanced=class_balanced)

        self.overlap_weight = overlap_weight
        self.overlap_idx = overlap_idx  # tuple containing the channel indices of pred, gt for overlap computation

    @staticmethod
    def overlap(pred_chan, gt_chan, pred, gt) -> torch.Tensor:
        pred = nn.Softmax(dim=1)(pred)
        # get the right channels from pred and gt tensors
        pred = pred[:, pred_chan, ...]
        gt = gt[:, gt_chan, ...]

        if gt.sum() != 0:
            return (pred * gt).sum() / gt.sum()

        return torch.tensor(0.)

    def forward(self, input, data) -> Dict[str, torch.Tensor]:
        target = data['labels']

        loss_dict = super().forward(input, data)

        # don't compute overlap if overlap_idx is set to None
        overlap_loss = self.overlap(*self.overlap_idx, input, target) if self.overlap_idx else torch.tensor(0.)

        loss_dict['overlap'] = self.overlap_weight * overlap_loss

        return loss_dict


@LOSS_REGISTRY.register()
class BoundaryLoss(nn.Module):
    def __init__(self, alpha_weight, normalization="softmax", class_labels=None, class_weight=None, class_balanced=False):
        super(BoundaryLoss, self).__init__()
        
        self.class_weight = None if not class_weight else torch.as_tensor(class_weight, dtype=torch.float)

        self.surface = SurfaceLoss(self.class_weight)
        self.surface_weight = surface_weight
        
        self.batch_count = 0
        self.alpha_weight = alpha_weight
        self.normalization = normalization
        self.dice = GeneralizedDiceLoss(normalization=normalization)
        
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
        if self.batch_count > 0 and (self.batch_count % 45 == 0):
            if self.alpha_weight >= 0.02:
                self.alpha_weight -= 0.01
            else:
                self.alpha_weight = 0.01

        self.batch_count += 1

    def forward(self, input, data):

        self.update_alpha_weight()

        target = data['labels']
        distms = data['dist_map']

        dice_loss = self.dice(input, target)
        dice_verbose = 1 - dice_loss.detach().cpu().numpy()

        if self.class_labels is not None:
            dice_labels_tuple = [i for i in zip(self.class_labels, dice_verbose)]
            dice_log = ["{} - {:.4f}, ".format(*i) for i in dice_labels_tuple]
        else:
            dice_log = ["{:.4f}, ".format(i) for i in dice_verbose]

        self.logger.info(("Dice: " + "{}" * len(dice_log)).format(*dice_log))

        if self.class_balanced:
            weights = self.get_class_balanced_weights(target)
            self.surface = SurfaceLoss(class_weights=weights)
            self.dice = DiceLoss(normalization=self.normalization, weight=weights)

        return {
            "dice": self.alpha_weight * dice_loss.sum(),
            "boundary": 1 - self.alpha_weight * self.surface(input, distms)
        }


@LOSS_REGISTRY.register()
class BoundaryBCELoss(BoundaryLoss):

    def __init__(self, bce_weight, dice_weight, surface_weight, normalization="softmax", class_labels=None, class_weight=None, class_balanced=False):
        
        super().__init__(dice_weight, surface_weight, normalization=normalization, class_labels=class_labels, 
            class_weight=class_weight, class_balanced=class_balanced)
        
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.bce_class_weight = self.class_weight.view(1, len(class_weight), 1, 1, 1).to(self.device)
        
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.bce_class_weight)
        self.bce_weight = bce_weight

    def forward(self, input, data):
        target = data['labels']

        loss_dict = super().forward(input, data)

        if self.class_balanced:
            weights = self.get_class_balanced_weights(target).view(1, -1, 1, 1, 1).to(self.device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=weights)

        loss_dict['bce'] = self.bce_weight * self.bce(input, target)

        return loss_dict


@LOSS_REGISTRY.register()
class BCEDiceSSIMLoss(nn.Module):

    def __init__(self, bce_weight, dice_weight, ssim_weight, class_weight=None, class_weight_loss="both",
                 normalization="sigmoid", class_labels=None):
        super(BCEDiceSSIMLoss, self).__init__()
        assert class_weight_loss in ["both", "bce", "dice"]
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.dice = DiceLoss(normalization=normalization)
        self.ssim_weight = ssim_weight
        self.ssim = SSIM()
        self.class_labels = class_labels
        self.logger = logging.getLogger(__name__)

        if class_weight is not None:
            class_weight = torch.as_tensor(class_weight, dtype=torch.float)
            if class_weight_loss == "dice":
                self.class_weight = class_weight
            elif class_weight_loss == "bce":
                self.bce = nn.BCEWithLogitsLoss(
                    pos_weight=self.class_weight.view(1, len(class_weight), 1, 1, 1)
                )
            else:
                # apply class weight to both dice and bce
                self.class_weight = class_weight
                self.bce = nn.BCEWithLogitsLoss(
                    pos_weight=self.class_weight.view(1, len(class_weight), 1, 1, 1)
                )
        else:
            self.class_weight = torch.as_tensor(1)

    def forward(self, input, target) -> Dict[str, torch.Tensor]:
        # dice
        dice_loss = self.dice(input, target)
        # get raw dice scores
        dice_verbose = 1 - dice_loss.detach().cpu().numpy()
        # apply per channel weighting to dice
        dice_loss *= self.class_weight.to(input.device)

        # bce
        self.bce.to(input.device)
        bce_loss = self.bce(input, target)

        # SSIM
        ssim_loss = []
        smax_input = nn.Softmax(dim=1)(input)
        for x, y in zip(smax_input[:, 1, ...].unsqueeze(1), target[:, 1, ...].unsqueeze(1)):
            ssim_loss.append(1 - self.ssim(x, y))

        ssim_loss = torch.tensor(ssim_loss).mean()

        if self.class_labels is not None:
            dice_labels_tuple = [i for i in zip(self.class_labels, dice_verbose)]
            dice_log = ["{} - {:.4f}, ".format(*i) for i in dice_labels_tuple]
        else:
            dice_log = ["{:.4f}, ".format(i) for i in dice_verbose]

        self.logger.info(("BCE: {:.8f} SSIM: {:.4f} Dice: " + "{}" * target.shape[1])
                         .format(bce_loss, 1 - ssim_loss, *dice_log))

        return {
            "bce": self.bce_weight * bce_loss,
            "dice": self.dice_weight * dice_loss.sum(),
            "ssim": self.ssim_weight * ssim_loss,
        }


@LOSS_REGISTRY.register()
class MultiLoss(nn.Module):
    """Linear combination of an arbitrary number of losses"""

    def __init__(self, losses: List[Tuple[str, dict]], weights: List[float] = None):
        """
        Args:
            losses: List of tuples where the first value is a loss registered in the LOSS REGISTRY
                    and the second value is a dictionary containing the params
                    E.g. [('DiceLoss',{}),
                          ('BCEWithLogitsLoss',{}),  # this loss is from torch.nn
                          ('GeneralizedDiceLoss',{'normalization':'softmax'}]
            weights: Weight applied to each loss specified in kwargs, must be same length as kwargs
        """
        super(MultiLoss, self).__init__()
        # if no weights specified then just set all weights to 1
        if not weights:
            weights = [1] * len(losses)
        self.register_buffer("weights", torch.as_tensor(weights))

        # initialize each loss from kwargs
        self.losses = {}
        for loss_name, params in losses:
            L = get_loss_criterion(loss_name)(**params)
            self.losses[loss_name] = L
            setattr(self, loss_name, L)

        self.logger = logging.getLogger(__name__)

    def forward(self, input, target) -> Dict[str, torch.Tensor]:
        results = {}
        # call forward for each loss
        for idx, loss_name in enumerate(self.losses):
            L = self.losses[loss_name](input, target)
            # check if result is a dict
            if type(L) is dict:
                # apply weighting to each value of dict and update results dict
                results = {
                    **results, **{k: self.weights[idx] * v for k, v in L.items()}
                }
            else:
                # apply weighting and update results dict
                results[loss_name] = self.weights[idx] * L

        return results


@LOSS_REGISTRY.register()
class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


@LOSS_REGISTRY.register()
class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


@LOSS_REGISTRY.register()
class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()


@LOSS_REGISTRY.register()
class SSIM(torch.nn.Module):
    # TODO: add creds to original author
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    @staticmethod
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


# HELPERS #
def get_loss_criterion(loss: str):
    return LOSS_REGISTRY.get(loss)


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

    dice = 2 * (intersect / denominator.clamp(min=epsilon))

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


# NOT-IN-USE CLASSES #
class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension if singleton
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


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


def get_optimizer(optim: str):
    return OPTIM_REGISTRY.get(optim)
