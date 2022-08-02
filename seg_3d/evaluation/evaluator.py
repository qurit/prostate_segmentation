import json
import logging
from typing import Callable, List, Tuple

import torch
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from seg_3d.data.dataset import ImageToImage3D
from seg_3d.evaluation.mask_visualizer import MaskVisualizer
from seg_3d.evaluation.metrics import MetricList
from seg_3d.utils.slice_builder import SliceBuilder


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
        self.mask_visualizer = None
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
            for idx, data_input in tqdm.tqdm(
                    enumerate(DataLoader(self.dataset, batch_size=1, num_workers=self.num_workers)),
                    total=len(self.dataset), desc="[evaluation progress =>]"):
                patient = data_input["patient"][0]
                sample = data_input["image"]  # shape is (batch, channel, depth, height, width)
                labels = data_input["gt_mask"].to(self.device)
                data = {'labels': labels,
                        'dist_map': data_input["dist_map"].to(self.device)}

                val_loss = []

                # runs the forward pass with autocasting if enabled
                with autocast(enabled=self.amp_enabled):

                    if self.slicer is not None:

                        preds = torch.zeros_like(labels).to(self.device)
                        norms = torch.zeros_like(labels).to(self.device)

                        for i in range(len(self.slicer.raw_slices)):
                            X, y = sample.squeeze(0)[self.slicer.raw_slices[i]], labels.squeeze(0)[self.slicer.label_slices[i]]
                            X, y = X.unsqueeze(0).to(self.device), y.unsqueeze(0).to(self.device)

                            y_hat = model(X).detach()
                            y_act = model.final_activation(y_hat)

                            u_prediction, u_index = self.slicer.remove_halo(y_act.squeeze(), self.slicer.raw_slices[i],
                                                                            sample.shape[2:], self.patch_halo)

                            u_index = (slice(0, 1, None), slice(0, 2, None)) + u_index[1:]

                            preds[u_index] += u_prediction
                            norms[u_index] += 1

                            if self.loss:
                                L = self.loss(y_hat, y)
                                if type(L) is dict:
                                    L = sum(L.values())
                                val_loss.append(L)

                        if self.loss:
                            val_loss = torch.stack(val_loss)
                            self.metric_list.results["val_loss"].append(torch.mean(val_loss).item())

                    else:
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
                        preds[:] = self.threshold_predictions(preds.squeeze(1))

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

        return {
            "inference": inference_dict,
            "metrics": {
                **self.metric_list.get_results(average=True)
            }
        }

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
