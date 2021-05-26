import os
import pickle
import time
import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count

from bladder_segmentation.mask_visualizer import MaskVisualizer


def plot_masks(patient):
    print(patient)
    patient_dict = pred_dict[patient]
    mask_visualizer.plot_mask_predictions(patient, np.arange(0, 128), np.arange(0, 128),
                                          patient_dict["image"].squeeze(), patient_dict["gt"].squeeze(),
                                          np.zeros_like(patient_dict["gt"]).squeeze(),
                                          patient_dict["preds"].squeeze())


# def visualize_preds(output_dir, crop_size, inference_fp="inference.pk"):
output_dir = "seg_3d/output/test-4"
inference_fp = "inference.pk"
crop_size = (128, 128)
with open(os.path.join(output_dir, inference_fp), "rb") as f:
    pred_dict = pickle.load(f, encoding='bytes')

plot_dir = os.path.join(output_dir, "mask_predictions")
mask_visualizer = MaskVisualizer(root_plot_dir=plot_dir, crop_size=crop_size, save_figs=True)

pool = Pool(processes=cpu_count())  # on 8 processors
pool.map(plot_masks, pred_dict)
pool.close()

if __name__ == '__main__':


    visualize_preds("seg_3d/output/test-4", (128, 128))
