import os

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from im_utils import centre_crop, dice_score


class MaskVisualizer:
    def __init__(self, algos: List[str] = None, crop_size=(100, 100), root_plot_dir="", save_figs=True):
        """
        Multi purpose class for plotting one or many per-frame mask predictions. First two subplots are the raw image
        and the ground truth bladder and/or tumor masks. This class should only be instantiated once when running pipeline.

        Args:
            algos: list of the algorithm names that will generate mask predictions
            crop_size: size of the center crop when plotting the masks
            root_plot_dir: root directory where plots are stored
            save_figs: option to save figures to disk or show them with plt.show()
        """
        if algos is None:
            algos = list()
        self.algos = algos
        self.crop_size = crop_size
        self.root_plot_dir = root_plot_dir
        self.save_figs = save_figs

        # compute number of subplots
        self.num_subplots = len(algos) + 2

        # configure color maps
        self.raw_img_cmap = "inferno"
        self.gt_cmap = "OrRd"
        self.pred_cmap = "Greens"

        # configure legend for predictions
        cmap = {1: [*np.asarray([62, 33, 10]) / 255, 1.0],
                2: [*np.asarray([129, 158, 131]) / 255, 1.0],
                3: [*np.asarray([251, 249, 241]) / 255, 1.],
                4: [*np.asarray([184, 126, 120]) / 255, 1.0]}
        labels = {1: 'TP', 2: 'FP', 3: 'TN', 4: 'FN'}
        self.pred_patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]

    def plot_mask_predictions(self, patient: str, bladder_frames_preds, bladder_frames_gt,
                              orig_img_3d, ground_truth_3d, tumor_mask_3d, pred_mask_3d_dict, dice_scores=None):
        """
        Generates a plot showing ground truth and prediction masks for each slice

        Args:
            patient: name of the patient
            bladder_frames_preds: list of the predicted indices part of the bladder frame range
            bladder_frames_gt: list of the ground truth indices part of the bladder frame range
            orig_img_3d: raw 3d image
            ground_truth_3d: bladder ground truth 3d mask
            tumor_mask_3d: tumor ground truth 3d mask
            pred_mask_3d_dict: predicted bladder 3d masks for each alg should be following structure {"alg_name": 3d_array}
            dice_scores: Option to show dice score across whole volume in title, needs to be list of floats
        """
        # check if directory to store patient plots exists
        patient_plot_dir = os.path.join(self.root_plot_dir, patient)
        if self.save_figs and not os.path.isdir(patient_plot_dir):
            os.makedirs(patient_plot_dir)

        dice_scores_title = ""
        if dice_scores is not None:
            dice_str_list = [str(round(dice, 4)) for dice in dice_scores]
            dice_scores_title = ", Dice Scores: " + " ".join(dice_str_list)

        # get full frame range
        frame_range = np.union1d(bladder_frames_preds, bladder_frames_gt).astype(int)

        # iterate through each slice
        for i, (orig_img_2d, gt_2d, tumor_2d) in enumerate(zip(orig_img_3d, ground_truth_3d, tumor_mask_3d)):
            # compute absolute index in scan
            abs_idx = i + frame_range[0]

            # check for any gt tumor and bladder overlap
            tumor_gt_overlap = gt_2d[tumor_2d == 1].sum()

            # configure legend for ground truth mask
            if tumor_gt_overlap != 0:
                # case where there is bladder, tumor, and overlap
                cmap = {1: list(cm.get_cmap(self.gt_cmap)(1.0)),
                        2: list(cm.get_cmap(self.gt_cmap)(0.33)),
                        3: list(cm.get_cmap(self.gt_cmap)(0.66))}
                labels = {1: 'Bladder', 2: 'Tumor', 3: 'Overlap'}
                gt_patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]

            elif tumor_2d.sum() != 0:
                if gt_2d.sum() == 0:
                    # case where there is only tumor
                    a = int(gt_2d.shape[0] / 2 - self.crop_size[0] / 2)
                    b = int(gt_2d.shape[1] / 2 - self.crop_size[1] / 2)
                    tumor_2d[a][b] += 3  # hack to keep tumor color consistent, puts a bright pixel at (0,0) of the image so it is minimally visible
                    gt_patches = [mpatches.Patch(color=list(cm.get_cmap(self.gt_cmap)(0.33)), label='Tumor')]
                else:
                    # case where there is both bladder and tumor
                    cmap = {1: list(cm.get_cmap(self.gt_cmap)(1.0)),
                            2: list(cm.get_cmap(self.gt_cmap)(0.33))}
                    labels = {1: 'Bladder', 2: 'Tumor'}
                    gt_patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]

            else:
                # case where there is only bladder
                gt_patches = [mpatches.Patch(color=list(cm.get_cmap(self.gt_cmap)(1.0)), label='Bladder')]

            # if only 1 alg make plot 1x3, otherwise make it two columns with as many rows required
            if len(self.algos) == 1:
                fig, axs = plt.subplots(1, 3, figsize=(10, 5))
                axs = np.expand_dims(axs, axis=0)
            else:
                fig, axs = plt.subplots(int(np.ceil(self.num_subplots / 2)), 2,
                                        figsize=(10, max(10, round(len(self.algos) / 4 * 10))))

            axs[0, 0].imshow(orig_img_2d, cmap=self.raw_img_cmap)
            axs[0, 0].set_title('Original')
            axs[0, 0].set_xlabel("x")
            axs[0, 0].set_ylabel("y")
            axs[0, 1].imshow(centre_crop(np.abs(gt_2d - tumor_2d / 3), self.crop_size), cmap=self.gt_cmap, interpolation='none', alpha=1.0)
            axs[0, 1].set_title('Ground Truth Mask\nGTSum: %.0f, B/ml: %.0f' % (gt_2d.sum(),
                                                                                orig_img_2d[gt_2d == 1].sum()))
            axs[0, 1].legend(handles=gt_patches, loc=4)  # loc=1 for upper right

            for idx, alg_name in enumerate(self.algos):
                idx += 2

                if len(self.algos) == 1:
                    a, b = 0, 2
                else:
                    # convert to 2d indices
                    a = int(idx / 2)
                    b = int(idx % 2)

                # get single slice from predicted bladder volume
                pred_mask_2d = pred_mask_3d_dict[alg_name][i]
                gt_2d[tumor_2d == 1] = 0
                slice_dice = 0. if pred_mask_2d.sum() == 0 else dice_score(gt_2d, pred_mask_2d)
                title = "%s Dice: %.04f\nPredSum: %.0f, B/ml: %.0f" % (alg_name, slice_dice, pred_mask_2d.sum(),
                                                                       orig_img_2d[pred_mask_2d == 1].sum())
                tumor_pred_overlap = pred_mask_2d[tumor_2d == 1].sum()
                if tumor_pred_overlap != 0:
                    title += "\nTumor Pred Overlap: %.0f" % tumor_pred_overlap

                axs[a, b].imshow(centre_crop(gt_2d, self.crop_size), cmap=self.gt_cmap, interpolation='none', alpha=1.0)
                axs[a, b].imshow(centre_crop(pred_mask_2d, self.crop_size), cmap=self.pred_cmap, interpolation='none', alpha=0.5)
                axs[a, b].set_title(title)
                axs[a, b].set_xticks([])
                axs[a, b].set_yticks([])

            fig.suptitle("%s Order: %.0f %s" % (patient, abs_idx, dice_scores_title))
            # add a text indicator if classifier misclassified frame
            if abs_idx not in bladder_frames_gt:
                # False Positive case
                fig.text(0.5, 0.925, "(False Positive frame)", ha="center", va="bottom", size="medium", color="red")
            elif abs_idx not in bladder_frames_preds:
                # False Negative case
                fig.text(0.5, 0.925, "(False Negative frame)", ha="center", va="bottom", size="medium", color="red")

            # add legend for the prediction masks
            plt.legend(handles=self.pred_patches, loc=4)  # loc=1 for upper right
            # adjust spacing
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)

            if self.save_figs:
                # save fig to disk
                fig.savefig(os.path.join(patient_plot_dir, str(abs_idx) + ".png"), format="png")
                plt.close(fig)

            else:
                plt.show()
                plt.close('all')
