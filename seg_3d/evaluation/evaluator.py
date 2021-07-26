import json
import logging
from typing import Callable, List, Tuple

import numpy as np
import tqdm
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from seg_3d.data.dataset import ImageToImage3D
from seg_3d.evaluation.metrics import MetricList


class Evaluator:
    def __init__(self, device: str, dataset: ImageToImage3D, metric_list: MetricList, loss: Callable = None,
                 thresholds: List[float] = None, amp_enabled: bool = False, patch_wise: Tuple[int] = None,
                 num_workers: int = 0):
        self.device = device
        self.dataset = dataset
        self.metric_list = metric_list
        self.loss = loss
        self.thresholds = thresholds
        self.amp_enabled = amp_enabled
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)

        if patch_wise is not None and np.prod(patch_wise) != 1:
            self.patch_wise = patch_wise
        else:
            self.patch_wise = None

    def evaluate(self, model):
        self.logger.info("Starting evaluation on dataset of size {}...".format(len(self.dataset)))
        self.metric_list.reset()
        if self.loss:
            self.metric_list.results["val_loss"] = []  # add an entry for val loss
        model.eval()

        inference_dict = {}
        with torch.no_grad():
            for idx, data_input in tqdm.tqdm(enumerate(DataLoader(self.dataset, batch_size=1, num_workers=self.num_workers)),
                                             total=len(self.dataset), desc="[evaluation progress =>]"):
                patient = data_input["patient"][0]
                sample = data_input["image"]  # shape is (batch, channel, depth, height, width)
                labels = data_input["gt_mask"]

                # divide sample into patches
                if self.patch_wise:
                    _, c, z, y, x = sample.shape
                    # remove slices along axial direction if necessary so it can be split into equal number patches
                    # num_slices = int(np.ceil(z / self.patch_wise[2]) * self.patch_wise[2] - self.patch_wise[2])
                    num_slices = int(np.ceil(z / self.patch_wise[2]))
                    sample, labels = sample[:, :, :num_slices], labels[:, :, :num_slices]

                    # reshape sample and labels, assumes tensors can be evenly divided in coronal and frontal direction by patch_wise
                    sample = sample.reshape(
                        np.prod(self.patch_wise), c, z // self.patch_wise[2], y // self.patch_wise[1], x // self.patch_wise[0]
                    )
                    _, c2, z, y, x = labels.shape
                    # orig_labels = labels.clone()
                    labels = torch.cat([
                        labels[:,cx,...].reshape(
                            np.prod(self.patch_wise), c, z // self.patch_wise[2], y // self.patch_wise[1], x // self.patch_wise[0])
                            for cx in range(c2)], dim=1)

                # runs the forward pass with autocasting if enabled
                with autocast(enabled=self.amp_enabled):
                    # iterate through each paired sample and label and get predictions from model
                    # feeding individual samples removes the condition of gpu memory fitting whole scan
                    preds = []
                    final_labels = []
                    val_loss = []
                    for X, y in zip(sample, labels):
                        X, y = X.unsqueeze(0).to(self.device), y.unsqueeze(0).to(self.device)
                        y_hat = model(X).detach()

                        if self.loss:
                            L = self.loss(y_hat, y)
                            if type(L) is dict:
                                L = sum(L.values())
                            val_loss.append(L)

                        preds.append(
                            # apply final activation on preds
                            model.final_activation(y_hat).squeeze().cpu()
                        )

                    preds = torch.stack(preds)
                    if self.loss:
                        val_loss = torch.stack(val_loss)
                        self.metric_list.results["val_loss"].append(torch.mean(val_loss).item())

                    # combine pred patches to a single sample
                    # if self.patch_wise:
                        # preds = preds.reshape(orig_labels.shape)
                        # labels = orig_labels

                    # apply thresholding if it is specified
                    if self.thresholds:
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
