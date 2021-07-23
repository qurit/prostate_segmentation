# original code from https://github.com/cosmic-cortex/pytorch-UNet/blob/master/unet/metrics.py
from typing import Callable, List, Dict

import numpy as np
import torch
from fvcore.common.registry import Registry
from sklearn.metrics import jaccard_score, f1_score
from skimage.metrics import structural_similarity, hausdorff_distance

from seg_3d.losses import compute_per_channel_dice

METRIC_REGISTRY = Registry('METRIC')
EPSILON = 1e-32


class MetricList:
    """
    Groups together metrics computed during evaluation for easy processing
    """
    def __init__(self, metrics: Dict[str, Callable], class_labels: List[str]):
        """
        Args:
            metrics: key is the name of the metric and value is the metric method that takes in two arguments
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
        self.results = {key: [] for key in self.metrics.keys()}

    def get_results_idx(self, idx):
        return {key: value[idx] for key, value in self.results.items()}

    def get_results(self, average=False):
        if not average:
            return self.results

        averaged_results = {}

        for key, value in self.results.items():
            if type(value[0]) == torch.Tensor and value[0].is_cuda:
                value = np.asarray([x.detach().cpu().numpy() for x in value]).mean(axis=0).tolist()
            else:
                value = np.asarray([np.asarray(x) for x in value]).mean(axis=0).tolist()

            if type(value) is list:
                for i in range(len(value)):
                    averaged_results[key + '/{}'.format(self.class_labels[i])] = value[i]
            else:
                averaged_results[key] = value

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
    return (pred[gt == 1]).sum() * 2.0 / (pred.sum() + gt.sum())


@METRIC_REGISTRY.register()
def classwise_dice_score(pred, gt):
    # FIXME: issue when AMP=True and pred and gt are on cpu, so put them on gpu
    return compute_per_channel_dice(pred.cuda(), gt.cuda(), epsilon=1e-6).cpu().numpy()


@METRIC_REGISTRY.register()
def argmax_dice_score(pred, gt):
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
    pred = pred.cpu().numpy().argmax(axis=1)
    gt = gt.cpu().numpy()

    pred[pred != 1] = 0

    gt = gt[:, 2, :, :, :]

    return (pred * gt).sum() / gt.shape[0]


@METRIC_REGISTRY.register()
def iou(pred, gt):
    return jaccard_score(gt.flatten(), pred.flatten().round())


@METRIC_REGISTRY.register()
def classwise_iou(pred, gt):
    """
    Args:
        pred: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    dims = (0, *range(2, len(pred.shape)))
    gt = gt.squeeze(1).long()
    gt = torch.zeros_like(pred).scatter_(1, gt[:, None, :], 1)
    intersection = pred * gt
    union = pred + gt - intersection
    return (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)


@METRIC_REGISTRY.register()
def f1(pred, gt):
    return f1_score(gt.flatten(), pred.flatten().round())


@METRIC_REGISTRY.register()
def classwise_f1(pred, gt):
    """
    Args:
        pred: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    gt = gt.long().squeeze(1)
    epsilon = 1e-20
    n_classes = pred.shape[1]

    pred = torch.argmax(pred, dim=1)
    true_positives = torch.tensor([((pred == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(pred == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    return (2 * (precision * recall) / (precision + recall))


@METRIC_REGISTRY.register()
def classwise_f1_v2(pred, gt):
    """
    Args:
        pred: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    gt = gt.long().squeeze(1)
    epsilon = 1e-20
    n_classes = pred.shape[1]

    pred = torch.argmax(pred, dim=1)
    true_positives = torch.tensor([((pred == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(pred == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    f1 = (2 * (precision * recall) / (precision + recall))
    return f1[1]


def get_metrics(config):
    return {
        metric: METRIC_REGISTRY.get(metric) for metric in config.TEST.EVAL_METRICS
    }
