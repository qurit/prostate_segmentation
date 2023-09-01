# original code from https://github.com/cosmic-cortex/pytorch-UNet/blob/master/unet/metrics.py
from typing import Callable, List, Dict

import numpy as np
import torch
from fvcore.common.registry import Registry
from skimage.metrics import structural_similarity, hausdorff_distance

from seg_3d.losses import compute_per_channel_dice


# nice way of keeping a 'record' of all the metrics, makes it easy to reference metrics inside a config yaml file
METRIC_REGISTRY = Registry('METRIC')
EPSILON = 1e-32


class MetricList:
    def __init__(self, metrics: Dict[str, Callable], class_labels: List[str]):
        """
        Groups together metrics computed during evaluation for easy processing

        Args:
            metrics: key is the name of the metric and value is the metric method that takes in two arguments (pred, gt)
            class_labels: labels for the classes, used to distinguish metric scores for different classes
        """
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: [] for key in self.metrics}
        self.class_labels = class_labels

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key].append(value(y_out, y_batch))

    def reset(self):
        self.results = {key: [] for key in self.metrics}

    def get_results_idx(self, idx):
        return {key: value[idx] for key, value in self.results.items()}

    def get_results(self, average=False):
        if not average:
            return self.results

        averaged_results = {}

        for key, value in self.results.items():
            # 0th dimension are patients and 1st dimension (if it exists) are ROIs
            if type(value[0]) == torch.Tensor and value[0].is_cuda:
                value = np.asarray([x.detach().cpu().numpy() for x in value])
            else:
                value = np.asarray([np.asarray(x) for x in value])

            # find mean
            value_mean = np.nanmean(value, axis=0).tolist()
            if type(value_mean) is list:
                for i in range(len(value_mean)):
                    averaged_results[key + '/{}'.format(self.class_labels[i])] = value_mean[i]
            else:
                averaged_results[key] = value_mean

            # hardcoding...
            if key in ["classwise_dice_score"]:
                for i in range(len(self.class_labels)):
                    if self.class_labels[i] in ["Bladder", "Tumor", "Inter"]:
                        averaged_results[key + '/{}'.format(self.class_labels[i]) + "_min"] = np.nanmin(value[:, i]).item()

                for i in range(len(self.class_labels)):
                    if self.class_labels[i] in ["Bladder", "Tumor", "Inter"]:
                        averaged_results[key + '/{}'.format(self.class_labels[i]) + "_std"] = np.std(value[:, i]).item()

        return averaged_results


@METRIC_REGISTRY.register()
def ssim(pred, gt):
    return [
        structural_similarity(im1, im2)
        for im1, im2 in zip(pred.squeeze().cpu().numpy(), gt.squeeze().cpu().numpy())
    ]


@METRIC_REGISTRY.register()
def hausdorff(pred, gt): 
    return [
        hausdorff_distance(im1, im2)
        for im1, im2 in zip(pred.squeeze().cpu().numpy(), gt.squeeze().cpu().numpy())
    ]


@METRIC_REGISTRY.register()
def dice_score(pred, gt):
    return (2.0 * (pred * gt).sum()) / (pred.sum() + gt.sum())


@METRIC_REGISTRY.register()
def classwise_dice_score(pred, gt):
    # FIXME: issue when AMP=True and pred and gt are on cpu, so put them on gpu
    return compute_per_channel_dice(pred.cuda(), gt.cuda(), epsilon=1e-6).cpu().numpy()


@METRIC_REGISTRY.register()
def argmax_dice_score(pred, gt):
    # NOTE: argmax after thresholding predictions does not make sense!
    pred = pred.cpu().numpy().argmax(axis=1)
    gt = gt.cpu().numpy()

    results = []
    # take care of background channel first
    pred_bg = pred.copy()
    pred_bg[pred_bg != 0] = -1
    pred_bg[pred_bg == 0] = 1
    pred_bg[pred_bg == -1] = 0

    results.append(
        dice_score(pred_bg, gt[:, 0, ...])
    )

    # iterate through the rest of the channels
    for i in range(1, gt.shape[1]):
        pred_i = pred.copy()
        pred_i[pred_i != i] = 0
        pred_i[pred_i == i] = 1
        results.append(
            dice_score(pred_i, gt[:, i, ...])
        )

    return results


@METRIC_REGISTRY.register()
def overlap(pred, gt):
    """Computes the overlap between predicted bladder and ground truth tumor"""
    # check if there are at least 3 channels
    if gt.shape[1] < 3:
       return np.NaN

    pred = pred.cpu().numpy().argmax(axis=1)
    gt = gt.cpu().numpy()

    pred[pred != 1] = 0  # hardcoded, get only predictions for bladder
    gt = gt[:, 2, :, :, :] # hardcoded, get only ground truth for tumor

    return (pred * gt).sum() / gt.sum()


@METRIC_REGISTRY.register()
def overlap_no_argmax(pred, gt):
    pred = pred[:, 1, ...]  # hardcoded
    gt = gt[:, 2, ...]  # hardcoded

    return (pred * gt).sum() / gt.sum()


def get_metrics(metrics: List[str]):
    return {
        metric: METRIC_REGISTRY.get(metric) for metric in metrics
    }
