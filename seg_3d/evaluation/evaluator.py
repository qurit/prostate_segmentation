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
        dummy_img = torch.ones((1,100,200,200))
        dummy_lab = torch.ones((2,100,200,200))
        self.slicer = SliceBuilder([dummy_img], [dummy_lab], (100,128,128),(100,72,72), None)

        # if patch_wise is not None and np.prod(patch_wise) != 1:
        if patch_wise is not None:
            self.patch_wise = patch_wise
        else:
            self.patch_wise = (1,1,1)

    def evaluate(self, model):
        self.logger.info("Starting evaluation on dataset of size {}...".format(len(self.dataset)))
        self.metric_list.reset()
        if self.loss:
            self.metric_list.results["val_loss"] = []  # add an entry for val loss
        model.eval()

        inference_dict = {}
        with torch.no_grad():
            
            f_dice_scores = []

            for idx, data_input in tqdm.tqdm(enumerate(DataLoader(self.dataset, batch_size=1, num_workers=self.num_workers)),
                                             total=len(self.dataset), desc="[evaluation progress =>]"):
                patient = data_input["patient"][0]
                sample = data_input["image"]  # shape is (batch, channel, depth, height, width)
                labels = data_input["gt_mask"]

                # runs the forward pass with autocasting if enabled
                with autocast(enabled=self.amp_enabled):
                    # if np.prod(self.patch_wise) != 1:
                    #     fX, fy = sample.clone().to(self.device), labels.clone().to(self.device)
                    #     y_full = model(fX).detach()
                    #     _ = self.loss(y_full, fy)
                    #     f_dice = argmax_dice_score(y_full, fy)
                    #     f_dice_scores.append(f_dice)

                    # iterate through each paired sample and label and get predictions from model
                    # feeding individual samples removes the condition of gpu memory fitting whole scan
                    preds = []
                    labels_list = []
                    val_loss = []
                    predmap = torch.zeros_like(labels).to(self.device)
                    normmap = torch.zeros_like(labels).to(self.device)

                    for i in range(len(self.slicer.raw_slices)):
                        X, y = sample.squeeze(0)[self.slicer.raw_slices[i]], labels.squeeze(0)[self.slicer.label_slices[i]]
                        X, y = X.unsqueeze(0).to(self.device), y.unsqueeze(0).to(self.device)
                        y_hat = model(X).detach()
                        
                        index = self.slicer.raw_slices[i]
                        # index = (slice(0,2,None), ) + index[1:]
                        y_act = model.final_activation(y_hat)
                        u_prediction, u_index = self.remove_halo(y_act.squeeze(), index, sample.shape[2:], (0,28,28))
                        u_index = (slice(0,1, None), slice(0,2,None)) + u_index[1:]

                        predmap[u_index] += u_prediction
                        normmap[u_index] += 1

                        if self.loss:
                            L = self.loss(y_hat, y)
                            if type(L) is dict:
                                L = sum(L.values())
                            val_loss.append(L)

                        # preds.append(
                        #     # apply final activation on preds
                        #     model.final_activation(y_hat).squeeze().cpu()
                        # )
                        # labels_list.append(y.squeeze())
                    
                    # preds = torch.stack(preds)
                    #labels = torch.stack(labels_list)
                    val_loss = torch.stack(val_loss)
                    self.metric_list.results["val_loss"].append(torch.mean(val_loss).item())

                    # apply thresholding if it is specified
                    if self.thresholds:
                        preds[:] = self.threshold_predictions(preds.squeeze(1))

                    self.metric_list(predmap, labels)

                    # print out results for patient
                    self.logger.info("results for patient {}:".format(patient))
                    print('YOOOOO', idx)
                    patient_metrics = self.metric_list.get_results_idx(idx)
                    for key in patient_metrics:
                        self.logger.info("{}: {}".format(key, patient_metrics[key]))
                    
                    # if np.prod(self.patch_wise) != 1:
                    #     self.logger.info("results for full scan {}:".format(f_dice))

                    inference_dict[patient] = {"gt": labels.cpu().numpy(),
                                               "preds": predmap,#preds.cpu().numpy(),
                                               "image": data_input["image"].numpy(),
                                               "metrics": patient_metrics}

        model.train()
        averaged_results = (self.metric_list.get_results(average=True))

        self.logger.info("Inference done! Mean metric scores:")
        self.logger.info(json.dumps(averaged_results, indent=4))
        # if np.prod(self.patch_wise) != 1:
        #     self.logger.info('argmax dice scores full scan: {}'.format(np.mean(np.asarray(f_dice_scores), axis=0)))

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
