import json
import logging
from typing import Callable, List

import tqdm
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from seg_3d.evaluation.metrics import MetricList


class Evaluator:
    def __init__(self, device, dataset, metric_list: MetricList, loss: Callable = None,
                 thresholds: List[float] = None, amp_enabled: bool = False):
        self.device = device
        self.dataset = dataset
        self.metric_list = metric_list
        self.loss = loss
        self.thresholds = thresholds
        self.amp_enabled = amp_enabled
        self.logger = logging.getLogger(__name__)

    def evaluate(self, model):
        self.logger.info("Starting evaluation on dataset of size {}...".format(self.dataset.__len__()))
        self.metric_list.reset()
        if self.loss:
            self.metric_list.results["val_loss"] = []  # add an entry for val loss
        model.eval()

        inference_dict = {}
        with torch.no_grad():
            for idx, data_input in tqdm.tqdm(enumerate(DataLoader(self.dataset, batch_size=1)),
                                             total=self.dataset.__len__(), desc="[evaluation progress =>]"):
                patient = data_input["patient"][0]
                sample = data_input["image"].to(self.device)
                labels = data_input["gt_mask"].squeeze(1).to(self.device)

                # runs the forward pass with autocasting if enabled
                with autocast(enabled=self.amp_enabled):
                    preds = model(sample).detach()

                    if self.loss is not None:
                        val_loss = self.loss(preds, labels)
                        if type(val_loss) is dict:
                            val_loss = sum(val_loss.values())
                        self.metric_list.results["val_loss"].append(val_loss.item())

                    # apply final activation on preds
                    preds = model.final_activation(preds)

                    # apply thresholding if it is specified
                    if self.thresholds is not None:
                        preds[:] = self.threshold_predictions(preds.squeeze(1))

                    self.metric_list(preds, labels)

                    # print out results for patient
                    self.logger.info("results for patient {}:".format(patient))
                    patient_metrics = self.metric_list.get_results_idx(idx)
                    for key in patient_metrics:
                        self.logger.info("{}: {}".format(key, patient_metrics[key]))

                    inference_dict[patient] = {"gt": labels.cpu().numpy(),
                                               "preds": preds.cpu().numpy(),
                                               "image": data_input["image"].numpy(),
                                               "metrics": patient_metrics}

        model.train()
        averaged_results = (self.metric_list.get_results(average=True))

        self.logger.info("Inference done! Mean metric scores:")
        self.logger.info(json.dumps(averaged_results, indent=4))

        # return inference dict and averaged results
        return {
            "inference": inference_dict,
            "metrics": {
                **self.metric_list.get_results(average=True)
            }
        }

    def threshold_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        # below approach only translates well to binary tasks
        # TODO: improve for multi class case
        new_preds = [
            torch.where(pred >= thres, pred, torch.zeros_like(pred))
            for thres, pred in zip(self.thresholds, preds)
        ]

        return torch.stack(new_preds)
