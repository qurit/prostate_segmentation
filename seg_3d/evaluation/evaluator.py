import json
import logging
from typing import Callable, List

import tqdm
import torch
from torch.utils.data import DataLoader


class Evaluator:
    def __init__(self, device, dataset, metric_list, loss: Callable = None, thresholds: List[float] = None):
        self.device = device
        self.dataset = dataset
        self.metric_list = metric_list
        self.loss = loss
        self.thresholds = thresholds
        self.logger = logging.getLogger(__name__)

    def evaluate(self, model, unsupervised=False):
        self.logger.info("Starting inference on dataset of size {}...".format(self.dataset.__len__()))
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

                preds = model(sample).detach()

                if self.loss:
                    if unsupervised:
                        loss_labels = (labels, sample)
                    else:
                        loss_labels = labels
                    self.metric_list.results["val_loss"].append(self.loss(preds, loss_labels).item())

                # apply final activation on preds
                preds = model.final_activation(preds)

                # apply thresholding if it is specified
                if self.thresholds is not None:
                    preds = self.threshold_predictions(preds)

                self.metric_list(preds, labels)

                # print out results for patient
                self.logger.info("results for patient {}:".format(patient))
                patient_metrics = self.metric_list.get_results_idx(idx)
                for key in patient_metrics:
                    self.logger.info("{}: {}".format(key, patient_metrics[key]))

                inference_dict[patient] = {"gt": labels.detach().cpu().numpy(),
                                           "preds": preds.detach().cpu().numpy(),
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

    def threshold_predictions(self, preds):
        return NotImplementedError  # TODO
