import json
import logging

import tqdm
import torch
from torch.utils.data import DataLoader


class Evaluator:
    def __init__(self, device, loss, dataset, metric_list):
        self.device = device
        self.loss = loss
        self.dataset = dataset
        self.metric_list = metric_list
        self.logger = logging.getLogger(__name__)

    def evaluate(self, model):
        self.logger.info("Starting inference on dataset of size {}...".format(self.dataset.__len__()))
        self.metric_list.reset()
        self.metric_list.results["val_loss"] = []  # add an entry for val loss
        model.eval()

        inference_dict = {}
        with torch.no_grad():
            for idx, data_input in tqdm.tqdm(enumerate(DataLoader(self.dataset, batch_size=1)),
                                             total=self.dataset.__len__(), desc="[evaluation progress =>]"):
                patient = data_input["patient"][0]
                sample = data_input["image"].to(self.device)
                labels = data_input["gt_mask"].squeeze(1).long().to(self.device)

                preds = model(sample).detach()
                self.metric_list.results["val_loss"].append(self.loss(preds, labels).item())

                # apply final activation on preds
                preds = model.final_activation(preds)
                print(model.final_activation)
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
