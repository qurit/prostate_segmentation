import json
import logging
from typing import Callable, List, Tuple

import numpy as np
import tqdm
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from seg_3d.data.dataset import ImageToImage3D
from seg_3d.evaluation.mask_visualizer import MaskVisualizer
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
        self.mask_visualizer = None
        self.logger = logging.getLogger(__name__)

        if patch_wise is not None and np.prod(patch_wise) != 1:
            self.patch_wise = patch_wise
        else:
            self.patch_wise = (1, 1, 1)

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

                # runs the forward pass with autocasting if enabled
                with autocast(enabled=self.amp_enabled):
                    # iterate through each paired sample and label and get predictions from model
                    # feeding individual samples removes the condition of gpu memory fitting whole scan
                    preds = []
                    labels_list = []
                    val_loss = []
                    for i in range(int(np.prod(self.patch_wise))):
                        X, y = self.get_patch(i, sample.squeeze(0), labels.squeeze(0))
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
                        labels_list.append(y.squeeze())

                    preds = torch.stack(preds)
                    labels = torch.stack(labels_list)
                    if self.loss:
                        val_loss = torch.stack(val_loss)
                        self.metric_list.results["val_loss"].append(torch.mean(val_loss).item())

                    # apply thresholding if it is specified
                    if self.thresholds:
                        preds[:] = self.threshold_predictions(preds.squeeze(0))

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

                    inference_dict[patient] = {"gt": labels,
                                               "preds": preds,
                                               "image": image,
                                               "metrics": patient_metrics}

                    # plot mask predictions if specified
                    if self.mask_visualizer:
                        self.mask_visualizer.plot_mask_predictions(
                            patient, image.squeeze(0), preds.squeeze(0), labels.squeeze(0), skip_bkg=True
                        )

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

    def get_patch(self, idx: int, image: torch.tensor, mask: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        if np.prod(self.patch_wise[:2]) == 1:  # do nothing if patch size is 1x1
            return image, mask

        # find the patch, indices for a 2x2 grid go like [0, 1; 2, 3]
        patch_idx = idx % np.prod(self.patch_wise[:2])
        # compute dimensions of patch
        m, n = (image.shape[2:] / np.asarray(self.patch_wise[:2])).astype(int)
        # find the row and column of the patch
        r, c = patch_idx // self.patch_wise[0], patch_idx % self.patch_wise[1]
        # handle case if patch size is a single column or row
        r = 0 if self.patch_wise[0] == 1 else r
        c = 0 if self.patch_wise[1] == 1 else c

        # gen slice objects, could have option to add padding for overlapping patches here
        s1 = slice(r * m, (r + 1) * m)
        s2 = slice(c * n, (c + 1) * n)

        # return the patch from image and mask
        return image[:, :, s1, s2], mask[:, :, s1, s2]

    def threshold_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        # below approach only translates well to binary tasks
        # TODO: improve for multi class case, maybe just argmax?
        new_preds = [
            torch.where(pred >= thres, torch.ones_like(pred), torch.zeros_like(pred))
            for thres, pred in zip(self.thresholds, preds)
        ]

        return torch.stack(new_preds)

    def set_mask_visualizer(self, class_labels, plot_dir):
        self.mask_visualizer = MaskVisualizer(class_labels=class_labels, root_plot_dir=plot_dir, save_figs=True)
