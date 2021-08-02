import json
import logging
from typing import Callable, List, Tuple

import numpy as np
import tqdm
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from seg_3d.data.dataset import ImageToImage3D, SliceBuilder
from seg_3d.evaluation.metrics import MetricList, argmax_dice_score


class Evaluator:
    def __init__(self, device: str, dataset: ImageToImage3D, metric_list: MetricList, loss: Callable = None,
                 thresholds: List[float] = None, amp_enabled: bool = False, num_workers: int = 0, 
                 patch_size: Tuple[int] = None, patch_stride: Tuple[int] = None, patch_halo: Tuple[int] = None,
                 patching_input_size: Tuple[int] = None, patching_label_size: Tuple[int] = None, **kwargs):

        self.device = device
        self.dataset = dataset
        self.metric_list = metric_list
        self.loss = loss
        self.thresholds = thresholds
        self.amp_enabled = amp_enabled
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_halo = patch_halo
        self.patch_input_size = patching_input_size
        self.patch_label_size = patching_label_size

        if self.patch_size is not None:
            dummy_img = torch.ones(self.patch_input_size)
            dummy_msk = torch.ones(self.patch_label_size)
            self.slicer = SliceBuilder([dummy_img], [dummy_msk], self.patch_size, self.patch_stride, None)
        else:
            self.slicer = None

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

                val_loss = []

                # runs the forward pass with autocasting if enabled
                with autocast(enabled=self.amp_enabled):

                    if self.slicer is not None:

                        preds = torch.zeros_like(labels).to(self.device)
                        norms = torch.zeros_like(labels).to(self.device)

                        patient_val_loss = []

                        for i in range(len(self.slicer.raw_slices)):
                            X, y = sample.squeeze(0)[self.slicer.raw_slices[i]], labels.squeeze(0)[self.slicer.label_slices[i]]
                            X, y = X.unsqueeze(0).to(self.device), y.unsqueeze(0).to(self.device)
                            
                            y_hat = model(X).detach()
                            y_act = model.final_activation(y_hat)

                            u_prediction, u_index = self.remove_halo(y_act.squeeze(), self.slicer.raw_slices[i], sample.shape[2:], self.patch_halo)
                            
                            u_index = (slice(0,1, None), slice(0,2,None)) + u_index[1:]

                            preds[u_index] += u_prediction
                            norms[u_index] += 1

                            if self.loss:
                                L = self.loss(y_hat, y)
                                if type(L) is dict:
                                    L = sum(L.values())
                                val_loss.append(L)

                        val_loss = torch.stack(val_loss)
                        self.metric_list.results["val_loss"].append(torch.mean(val_loss).item())

                    else:

                        preds = model(sample).detach()

                        if self.loss:
                            L = self.loss(preds, labels.to(self.device))
                            if type(L) is dict:
                                L = sum(L.values())

                        # apply final activation on preds
                        preds = model.final_activation(preds)
                    
                        self.metric_list.results["val_loss"].append(L)

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
    
    @staticmethod
    def remove_halo(patch, index, shape, patch_halo):
        """
        Remove `pad_width` voxels around the edges of a given patch.
        """
        assert len(patch_halo) == 3

        def _new_slices(slicing, max_size, pad):
            if slicing.start == 0:
                p_start = 0
                i_start = 0
            else:
                p_start = pad
                i_start = slicing.start + pad

            if slicing.stop == max_size:
                p_stop = None
                i_stop = max_size
            else:
                p_stop = -pad if pad != 0 else 1
                i_stop = slicing.stop - pad

            return slice(p_start, p_stop), slice(i_start, i_stop)

        D, H, W = shape

        i_c, i_z, i_y, i_x = index
        p_c = slice(0, patch.shape[0])

        p_z, i_z = _new_slices(i_z, D, patch_halo[0])
        p_y, i_y = _new_slices(i_y, H, patch_halo[1])
        p_x, i_x = _new_slices(i_x, W, patch_halo[2])

        patch_index = (p_c, p_z, p_y, p_x)
        index = (i_c, i_z, i_y, i_x)
        return patch[patch_index], index
