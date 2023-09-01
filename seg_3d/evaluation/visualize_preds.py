import json
import os
import pickle
from typing import List, Tuple

import numpy as np
import tqdm

from seg_3d.evaluation.mask_visualizer import MaskVisualizer


def visualize_preds(inference_fp: str, class_labels: List[str], crop_size: Tuple, plane: str = "", save_figs=False):
    """
    Runs the mask visualizer for a set of predictions stored on disk.

    Args:
        inference_fp: file path to the inference pickle file storing the predictions
        class_labels: the classes (e.g. Bladder, Tumor) to include in the figure
        crop_size: size of the center crop when plotting the masks
        plane: the anatomical plane in which to do slicing
        save_figs: option to save figures to disk or show them with plt.show()
    """
    # load the predictions stored on disk
    with open(os.path.join(inference_fp), "rb") as f:
        pred_dict = pickle.load(f, encoding="bytes")

    # initialize an instance of the mask visualizer
    mask_visualizer = MaskVisualizer(class_labels=class_labels,
                                     root_plot_dir=os.path.join(os.path.dirname(inference_fp), "masks", plane),
                                     crop_size=crop_size,
                                     save_figs=save_figs)

    # iterate through each patient
    for patient in tqdm.tqdm(pred_dict, desc="[visualize predictions progress =>]"):
        patient_dict = pred_dict[patient]
        scores = {k: v.tolist() if type(v) is np.ndarray else v for k, v in patient_dict["metrics"].items()}

        print("\n", patient, "\n", scores)

        mask_visualizer.plot_mask_predictions(patient,
                                              patient_dict["image"].squeeze(0),
                                            #   alternatively, pass in original image:
                                            #   patient_dict["orig_image"].squeeze(0),
                                              patient_dict["preds"].squeeze(0),
                                            #   alternatively, pass in empty predictions:
                                            #   np.zeros_like(patient_dict["gt"].squeeze(0)),
                                              patient_dict["gt"].squeeze(0),
                                              skip_bkg=True,           # first channel is background so skip it
                                              gt_overlay=False,        # option to overlay the ground truth with prediction
                                              plane=plane,             # anatominal plane
                                              show_slice_scores=True   # option to show slice dice scores in figure
                                              )


if __name__ == '__main__':
    """
    Run demo via:
    `python -m seg_3d.evaluation.visualize_preds`
    """
    for p in ['tran', 'sag', 'cor']:
        visualize_preds(# some items in the pickle file may have been stored while being on the gpu
                        inference_fp="seg_3d/output/FINAL_PET_bladder_model/1/inference.pk",
                        class_labels=["Bladder", "Tumor"],
                        crop_size=(128, 128),
                        save_figs=True,
                        plane=p
                        )
