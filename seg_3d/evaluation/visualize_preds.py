import json
import os
import pickle

import numpy as np
import tqdm

from seg_3d.evaluation.mask_visualizer import MaskVisualizer


def visualize_preds(output_dir, class_labels, crop_size, inference_fp="test_inference.pk"):
    with open(os.path.join(output_dir, inference_fp), "rb") as f:
        pred_dict = pickle.load(f, encoding='bytes')

    plot_dir = os.path.join(output_dir, "mask_predictions")
    mask_visualizer = MaskVisualizer(class_labels=class_labels, root_plot_dir=plot_dir, crop_size=crop_size,
                                     save_figs=False)

    # iterate through each patient
    for patient in tqdm.tqdm(pred_dict, desc="[visualize predictions progress =>]"):
        patient_dict = pred_dict[patient]
        scores = {k: v.tolist() if type(v) is np.ndarray else v for k, v in patient_dict["metrics"].items()}
        print("\n", patient, "\n", json.dumps(scores, indent=4))

        mask_visualizer.plot_mask_predictions(
            patient, patient_dict["image"].squeeze(0), patient_dict["preds"].squeeze(0), patient_dict["gt"].squeeze(0),
            skip_bkg=True, gt_overlay=True
        )


if __name__ == '__main__':
    # visualize_preds("../output/prostate-runs/baseline", class_labels=["Inter"], crop_size=None)
    # visualize_preds("../output/bladder-runs/bladder-tumor-class-weight-0,1,1-both", class_labels=["Bladder", "Tumor"], crop_size=None)
    visualize_preds("../output/slice-392-even-later-lr-drop-fmaps-32-amp/eval-with-sigmoid",
                    class_labels=["Inter", "Bladder", "Tumor"], crop_size=None)  # "TURP urethra", "R seminal", "L seminal"
    # #
    # TODO: make it easy to run from command line
    # TODO: easy way to add multiple predictions from algorithms to compare
