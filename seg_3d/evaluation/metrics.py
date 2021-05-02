# original code from https://github.com/cosmic-cortex/pytorch-UNet/blob/master/unet/metrics.py
import numpy as np
import torch

from seg_3d.losses import compute_per_channel_dice

from detectron2.utils.registry import Registry
from sklearn.metrics import jaccard_score, f1_score

METRIC_REGISTRY = Registry('METRIC')
EPSILON = 1e-32


class MetricList:
    def __init__(self, metrics, class_labels):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: [] for key in self.metrics.keys()}
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
        else:
            averaged_results = {}

            for key, value in self.results.items():
                if type(value[0]) == torch.Tensor and value[0].is_cuda:
                    value = np.asarray([x.detach().cpu().numpy() for x in value]).mean(axis=0).tolist()
                else:
                    value = np.asarray([np.asarray(x) for x in value]).mean(axis=0).tolist()
                
                if len(value) > 1:
                    for i in range(len(values)):
                        averaged_results[key+'-{}'.format(class_labels[i])] = values[i]
                else:
                    averaged_results[key] = value

            return averaged_results


@METRIC_REGISTRY.register()
def dice_score(pred, gt, round=True):
    if round:
        pred = pred.round()  # threshold pred
    
    return (pred[gt == 1]).sum() * 2.0 / (pred.sum() + gt.sum())


@METRIC_REGISTRY.register()
def classwise_dice_score(pred, gt):
    return compute_per_channel_dice(pred, gt, epsilon=1e-6).detach().cpu().numpy()

@METRIC_REGISTRY.register()
def bladder_dice_score(pred, gt):
    return compute_per_channel_dice(pred, gt, epsilon=1e-6)[1]

@METRIC_REGISTRY.register()
def argmax_dice_score(pred, gt):
    pred = pred.detach().cpu().numpy().argmax(axis=1)
    gt = gt.detach().cpu().numpy()

    pred_bg, pred_blad, pred_tum = pred.copy(), pred.copy(), pred.copy()
    pred_bg[pred_bg != 0] = -1
    pred_bg[pred_bg == 0] = 1
    pred_bg[pred_bg == -1] = 0

    pred_blad[pred_blad != 1] = 0

    pred_tum[pred_tum != 2] = 0
    pred_tum[pred_tum == 2] = 1

    gt_bg = gt[:, 0, :, :, :]
    gt_blad = gt[:, 1, :, :, :]
    gt_tum = gt[:, 2, :, :, :]

    dice_bg = dice_score(pred_bg, gt_bg, round=False)
    dice_blad = dice_score(pred_blad, gt_blad, round=False)
    dice_tum = dice_score(pred_tum, gt_tum, round=False)

    return [dice_bg, dice_blad, dice_tum]


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
    return {metric: METRIC_REGISTRY.get(metric) for metric in config.TEST.EVAL_METRICS}
