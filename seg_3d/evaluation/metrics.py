# original code from https://github.com/cosmic-cortex/pytorch-UNet/blob/master/unet/metrics.py
import numpy as np
import torch

from seg_3d.losses import compute_per_channel_dice

from detectron2.utils.registry import Registry
from sklearn.metrics import jaccard_score, f1_score

METRIC_REGISTRY = Registry('METRIC')
EPSILON = 1e-32


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: [] for key in self.metrics.keys()}

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
        else:
            return {key: float(np.mean(value)) for key, value in self.results.items()}


@METRIC_REGISTRY.register()
def dice_score(pred, gt):
    pred = pred.round()  # threshold pred
    return (pred[gt == 1]).sum() * 2.0 / (pred.sum() + gt.sum())


@METRIC_REGISTRY.register()
def classwise_dice_score(pred, gt):
    return torch.mean(compute_per_channel_dice(pred.unsqueeze(0), gt.unsqueeze(0), epsilon=1e-6))


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
    intersection = pred*gt
    union = pred + gt - intersection
    return ((intersection.sum(dim=dims).float() + EPSILON) / (
        union.sum(dim=dims) + EPSILON)).item()


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
    return (2 * (precision * recall) / (precision + recall)).item()


def get_metrics(config):
    return {metric: METRIC_REGISTRY.get(metric) for metric in config.TEST.EVAL_METRICS}
