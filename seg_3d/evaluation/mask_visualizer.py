import os
from collections import OrderedDict

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple

from utils import centre_crop, dice_score


# define mapping of roi to cmap
# all sequential cmaps here https://matplotlib.org/stable/tutorials/colors/colormaps.html#sequential
# this dict also specifies the order in which rois are plotted e.g. Inter is plotted first, then Bladder
cmap_roi_map = OrderedDict({
    "Inter": "Greens",
    "Bladder": "Reds",
    "Tumor": "Purples",
    "TURP urethra": "Greys",
    "R seminal": "Oranges",
    "L seminal": "Blues"
})


class MaskVisualizer:
    def __init__(self, class_labels: List[str], algos: List[str] = None, crop_size: Tuple = None,
                 root_plot_dir: str = "", fig_grid: Tuple = None, fig_size: Tuple = None, save_figs: bool = True):
        """
        Multi purpose class for plotting one or many per-frame mask predictions. First two subplots are the raw image
        and the ground truth masks. This class should only be instantiated once when running pipeline.

        Args:
            class_labels: list of names of the plotted region of interests
            algos: list of the algorithm names that will generate mask predictions
            crop_size: size of the center crop when plotting the masks
            fig_grid: number of rows and columns for the figure grid
            fig_size: dimensions of the figure
            root_plot_dir: root directory where plots are stored
            save_figs: option to save figures to disk or show them with plt.show()
        """
        if algos is None:
            algos = [""]
        self.algos = algos
        self.class_labels = class_labels
        assert len(set(self.class_labels) - set(cmap_roi_map.keys())) == 0,\
            "a mapping from class to color map needs to be defined for all class labels"
        self.crop_size = crop_size
        self.root_plot_dir = root_plot_dir
        self.fig_grid = fig_grid
        self.fig_size = fig_size
        self.save_figs = save_figs

        # configure color maps
        self.sample_cmap = "inferno"
        self.pred_cmap_dict = OrderedDict({label: cmap_roi_map[label] for label in class_labels})
        self.plot_ordering = {label: idx for idx, label in enumerate(cmap_roi_map) if label in self.pred_cmap_dict}

    def plot_mask_predictions(
            self, patient: str, sample: np.ndarray, pred_dict: Dict[str, np.ndarray] or np.ndarray,
            gt: np.ndarray, gt_overlay: bool = False, skip_bkg: bool = True, show_slice_scores: bool = True,
            plane: str = None,
    ) -> None:
        """
        Generates a plot showing original sample, ground truth and prediction masks for each slice

        Args:
            patient: name of the patient
            sample: raw sample of shape (D, H, W) or (C, D, H, W), channel here are the different modalities
            gt: ground truth mask of shape (C, D, H, W), channel here are the different classes
            pred_dict: predicted masks for each alg where keys are alg names and items are preds of shape (C, D, H, W)
            gt_overlay: option to overlay the ground truth mask with the prediction
            skip_bkg: option to skip the first channel in gt and pred_dict which is the background channel
            show_slice_scores: option to show dice score and pixel sum for each slice
            plane: the anatomical plane in which to do slicing, options are 'tra' (default), 'sag' or 'cor'
        """
        # handle case where only have 1 set of predictions
        if not isinstance(pred_dict, dict) and len(self.algos) == 1 and isinstance(pred_dict, np.ndarray):
            pred_dict = {self.algos[0]: pred_dict}

        # make sure pred and gt are dimension 4
        if len(gt.shape) != 4:
            gt = np.expand_dims(gt, 0)
        pred_dict = {k: np.expand_dims(v, 0) if len(v.shape) != 4 else v for k, v in pred_dict.items()}

        if skip_bkg and len(gt) > 1:
            gt = gt[1:]
            pred_dict = {k: v[1:] for k, v in pred_dict.items()}

        if not self.class_labels:
            self.class_labels = [""] * len(gt)

        # handle case for viewing slices in specified plane
        if plane == "cor":  # Coronal
            sample = np.rot90(np.swapaxes(sample, -3, -2), k=2, axes=(-2, -1))
            pred_dict = {k: np.rot90(np.swapaxes(v, -3, -2), k=2, axes=(-2, -1)) for k, v in pred_dict.items()}
            gt = np.rot90(np.swapaxes(gt, -3, -2), k=2, axes=(-2, -1))
        elif plane == "sag":  # Sagittal
            sample = np.rot90(np.swapaxes(sample, -3, -1), k=1, axes=(-2, -1))
            pred_dict = {k: np.rot90(np.swapaxes(v, -3, -1), k=1, axes=(-2, -1)) for k, v in pred_dict.items()}
            gt = np.rot90(np.swapaxes(gt, -3, -1), k=1, axes=(-2, -1))
        else:
            pass  # nothing to do for Axial / Transverse plane

        # check sample, pred, and gt all have same depth size
        num_slices = sample.shape[0] if len(sample.shape) == 3 else sample.shape[1]
        assert num_slices == gt.shape[1] == pred_dict[self.algos[0]].shape[1]

        # check self.algos matches algos in pred_dict
        for a in self.algos:
            assert a in pred_dict, "pred_dict keys must exist in self.algos"

        # get the range of frames where gt mask and predictions are nonzero
        gt_range = []
        for y in gt:
            gt_range.extend(
                [i for i, y_i in enumerate(y) if y_i.sum() != 0]
            )
        pred_range = []
        for preds in pred_dict.values():
            for y_hat in preds:
                # only add slices which have at least some pixels that are 'bright'
                pred_range.extend(
                    [i for i, y_hat_i in enumerate(y_hat) if y_hat_i[y_hat_i > 0.5].sum() >= 3]
                )
        # select the range of slices where gt mask and predictions are (mostly) nonzero
        num_slices = np.union1d(gt_range, pred_range).astype(int).tolist()

        # check if directory to store plots exists
        patient_plot_dir = os.path.join(self.root_plot_dir, patient)
        if self.save_figs and not os.path.isdir(patient_plot_dir):
            os.makedirs(patient_plot_dir)

        # make sure figure size and grid layout are defined
        self.check_fig_setup(sample)

        # iterate through each slice
        for i in num_slices:
            # initialize new figure and axes
            fig, axs = plt.subplots(*self.fig_grid, figsize=self.fig_size)
            if len(axs.shape) == 1:
                axs = np.expand_dims(axs, axis=0)  # make sure axs is 2d

            # plot sample which can have multiple modalities
            for j, X in enumerate(sample):
                # convert to 2d indices
                a, b = int(j / axs.shape[1]), int(j % axs.shape[1])

                # show sample and annotate plot
                im = axs[a, b].imshow(centre_crop(X[i], self.crop_size), cmap=self.sample_cmap)
                cb = plt.colorbar(im, ax=axs[a, b], shrink=0.7)
                axs[a, b].set_title("Sample")
                axs[a, b].set_xlabel("x")
                axs[a, b].set_ylabel("y")

            # get 2d indices for ground truth mask
            a, b = int(len(sample) / axs.shape[1]), int(len(sample) % axs.shape[1])
            gt_images = []  # make a list to keep track of the images plotted since will reuse them in next for loop
            gt_patches = []  # for legend

            # plot ground truth mask for each roi
            for j, (y, label) in enumerate(zip(gt, self.class_labels)):
                y = centre_crop(y[i], self.crop_size)

                # init RGBA array
                gt_im = np.zeros(y.shape + (4,))
                if y.sum() != 0:
                    # set the transparency to 0 for pixels which have a prediction of 0
                    gt_im[:, :, 3] = y
                    # fill in the RGB values
                    # multiply the values by some factor (e.g. 0.5) so they
                    # are a shade different than the mask predictions
                    gt_im[:, :, :3] = cm.get_cmap(self.pred_cmap_dict[label])(y * 0.5)[:, :, :3]

                    # add sum to legend if specified
                    leg_label = label + ": {:.0f}".format(y.sum()) if show_slice_scores else label

                    # add patch for legend
                    gt_patches.append(
                        mpatches.Patch(color=list(cm.get_cmap(self.pred_cmap_dict[label])(0.5)), label=leg_label)
                    )
                gt_images.append(gt_im)

                # show ground truth
                axs[a, b].imshow(gt_im, interpolation="none", alpha=1.0, zorder=self.plot_ordering[label])

            # add title and legend
            axs[a, b].set_title("Ground Truth Mask")
            axs[a, b].legend(handles=gt_patches, loc=1)  # loc=1 for upper right

            pred_patches = []  # for legend
            # iterate through each algorithm and its set of predictions
            for j, alg_name in enumerate(self.algos):
                j += len(sample) + 1

                # get single slice from prediction
                pred = pred_dict[alg_name][:, i]

                # convert to 2d indices
                a, b = int(j / axs.shape[1]), int(j % axs.shape[1])
                title = "Predicted Mask %s" % alg_name
                axs[a, b].set_title(title)
                axs[a, b].set_xticks([])
                axs[a, b].set_yticks([])

                # show gt in the background if specified
                pred_alpha = 1.
                if gt_overlay:
                    pred_alpha = .8
                    for gt_im in gt_images:
                        # gt_im[:, :, :3] *= 0  # make gt all black
                        axs[a, b].imshow(gt_im, interpolation="none", alpha=0.5)

                # iterate through each channel in the pred
                for _, (y_hat, y, label) in enumerate(zip(pred, gt, self.class_labels)):
                    # apply centre cropping as a post processing step
                    y_hat = centre_crop(y_hat, self.crop_size)
                    y = centre_crop(y[i], self.crop_size)

                    # make RGBA array for pred
                    pred_im = np.zeros(y_hat.shape + (4,))
                    if y_hat[y_hat > 0.5].sum() != 0:
                        pred_im[:, :, 3] = y_hat
                        pred_im[:, :, :3] = cm.get_cmap(self.pred_cmap_dict[label])(y_hat)[:, :, :3]

                        # add dice and sum to legend if specified
                        if show_slice_scores:
                            leg_label = label + ": {:.0f}, {:.4f}".format(
                                np.ceil(y_hat[y_hat > 0.5]).sum(), dice_score(y, y_hat)
                            )
                        else:
                            leg_label = label

                        pred_patches.append(
                            mpatches.Patch(color=list(cm.get_cmap(self.pred_cmap_dict[label])(1.)), label=leg_label)
                        )

                    # show pred
                    axs[a, b].imshow(pred_im, interpolation="none", alpha=pred_alpha, zorder=self.plot_ordering[label])

            # set aspect to equal for all subplots
            for ax in axs.flatten():
                ax.set_aspect("equal")

            fig.suptitle("%s Order: %.0f" % (patient, i))
            # add legend for the prediction masks
            plt.legend(handles=pred_patches, loc=1)
            # adjust spacing
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)

            if self.save_figs:
                # save fig to disk
                fig.savefig(os.path.join(patient_plot_dir, str(i) + ".png"), format="png")
                plt.close(fig)

            else:
                plt.show()
                plt.close('all')

    def check_fig_setup(self, sample):
        # compute figure grid layout if not specified
        if not self.fig_grid:
            if len(self.algos) == 1:
                # if only 1 alg and 1 modality then make fig 1x3
                if sample.shape[0] == 1:
                    self.fig_grid = (1, 3)
                    if not self.fig_size:
                        self.fig_size = (10, 5)

                # if there are two modalities then make fig 2x2
                elif sample.shape[0] == 2:
                    self.fig_grid = (2, 2)
                    if not self.fig_size:
                        self.fig_size = (10, 10)

            # make fig n x 2 where n is the number of necessary rows
            else:
                self.fig_grid = (int(np.ceil((len(self.algos) + sample.shape[0] + 1) / 2)), 2)
                if not self.fig_size:
                    self.fig_size = (10, max(10, round(len(self.algos) / 4 * 10)))
        # make sure fig size is specified
        if not self.fig_size:
            self.fig_size = (self.fig_grid[1] * 4, self.fig_grid[0] * 4)
