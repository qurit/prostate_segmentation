import json
import logging
import os
from typing import Callable, List, Tuple

import torch
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from seg_3d.data.dataset import ImageToImage3D
from seg_3d.evaluation.mask_visualizer import MaskVisualizer
from seg_3d.evaluation.metrics import MetricList


class Evaluator:
    def __init__(self, device: str, dataset: ImageToImage3D, metric_list: MetricList, loss: Callable = None,
                 thresholds: List[float] = None, amp_enabled: bool = False, num_workers: int = 0):
        """
        Class for evaluating a model.

        Args:
            device: the cpu or gpu device on which to load tensors
            dataset: an instance of ImageToImage3D for loading samples from dataset
            metric_list: an instance of MetricList to specify list of metrics to compute
            loss: optional arg for tracking validation loss
            thresholds: optional arg to threshold continuous valued predictions
            amp_enabled: whether not we are in AMP mode
            num_workers: can distribute evaluation across multiple workers
        """
        self.device = device
        self.dataset = dataset
        self.metric_list = metric_list
        self.loss = loss
        self.thresholds = thresholds
        self.amp_enabled = amp_enabled
        self.num_workers = num_workers
        self.mask_visualizer = None
        self.logger = logging.getLogger(__name__)

    def evaluate(self, model):
        self.logger.info("Starting evaluation on dataset of size {}...".format(len(self.dataset)))
        self.metric_list.reset()
        if self.loss:
            self.metric_list.results["val_loss"] = []  # add an entry for val loss
        model.eval()

        inference_dict = {}
        with torch.no_grad():
            for idx, data_input in tqdm.tqdm(
                    enumerate(DataLoader(self.dataset, batch_size=1, num_workers=self.num_workers)),
                    total=len(self.dataset), desc="[evaluation progress =>]"):
                patient = data_input["patient"][0]
                sample = data_input["image"]  # shape is (batch, channel, depth, height, width)
                labels = data_input["gt_mask"].to(self.device)
                data = {"labels": labels,
                        "dist_map": data_input["dist_map"].to(self.device)}

                # runs the forward pass with autocasting if enabled
                with autocast(enabled=self.amp_enabled):
                    preds = model(sample).detach()

                    if self.loss:
                        L = self.loss(preds, data)
                        if type(L) is dict:
                            L = sum(L.values())
                        self.metric_list.results["val_loss"].append(L)

                    # apply final activation on preds
                    preds = model.final_activation(preds)

                    # apply thresholding if it is specified
                    if self.thresholds:
                        preds[:] = self.threshold_predictions(preds.squeeze())

                    # make sure pred and labels have same number of channels
                    if preds.shape[1] < labels.shape[1]:
                        shape = list(labels.shape)
                        shape[1] = labels.shape[1] - preds.shape[1]
                        preds = torch.cat((preds, torch.zeros(shape).to(self.device)), dim=1)

                    # get scores for all metrics
                    self.metric_list(preds, labels)

                    # print out results for patient
                    self.logger.info("results for patient {}:".format(patient))
                    patient_metrics = self.metric_list.get_results_idx(idx)
                    for key in patient_metrics:
                        self.logger.info("{}: {}".format(key, patient_metrics[key]))

                    # make sure all tensors are on cpu and convert to npy arrays
                    labels = labels.cpu().numpy()
                    preds = preds.cpu().numpy()
                    image = data_input["image"].cpu().numpy()
                    orig_image = data_input["orig_image"].cpu().numpy()

                    inference_dict[patient] = {"gt": labels,
                                               "preds": preds,
                                               "image": image,
                                               "orig_image": orig_image,
                                               "metrics": patient_metrics}

                    # plot mask predictions if specified
                    if self.mask_visualizer:
                        for plane in ["tra", "cor", "sag"]:
                            self.mask_visualizer.root_plot_dir = os.path.join(os.path.dirname(self.mask_visualizer.root_plot_dir), plane)
                            self.mask_visualizer.plot_mask_predictions(
                                patient, image.squeeze(0), preds.squeeze(0), labels.squeeze(0),
                                skip_bkg=True, gt_overlay=False, plane=plane
                            )

        model.train()
        averaged_results = (self.metric_list.get_results(average=True))

        self.logger.info("Inference done! Mean metric scores:")
        self.logger.info(json.dumps(averaged_results, indent=4))

        return {
            "inference": inference_dict,
            "metrics": {
                **self.metric_list.get_results(average=True)
            }
        }

    def threshold_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        new_preds = []
        for thres, pred in zip(self.thresholds, preds):
            if thres is None:
                new_preds.append(pred)
                continue
            new_preds.append(
                torch.where(pred >= thres, torch.ones_like(pred), torch.zeros_like(pred))
            )

        return torch.stack(new_preds)

    def set_mask_visualizer(self, class_labels, plot_dir):
        self.mask_visualizer = MaskVisualizer(class_labels=class_labels, root_plot_dir=plot_dir, save_figs=True)
