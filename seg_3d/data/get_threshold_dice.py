import json
import pickle
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from seg_3d.data.dataset import ImageToImage3D, JointTransform3D
from seg_3d.evaluation.mask_visualizer import MaskVisualizer
from utils import dice_score


# load dataset
dataset = ImageToImage3D(
    dataset_path='data/image_dataset',
    modality_roi_map=[{'CT': ['Inter', 'Threshold']}, {'PT': ['Bladder', 'Tumor', 'Tumor2', 'Tumor3']}],
    class_labels=['Background', 'Bladder', 'Tumor', 'Inter', 'Threshold'],
    slice_shape=(512, 512),
    joint_transform=None
)
print(dataset.patient_keys, '\n', len(dataset.patient_keys))

root_dir = 'seg_3d/output/zthreshold'
mask_visualizer = MaskVisualizer(class_labels=['Inter'], root_plot_dir=os.path.join(root_dir, 'masks'),
                                 crop_size=(200, 200), save_figs=True)

scores = {}
for i in tqdm(range(len(dataset))):
    sample = dataset[i]
    patient = sample['patient']
    print('\n', patient)
    image, mask = sample['image'].numpy(), sample['gt_mask'].numpy()
    gt, pred = mask[3], mask[4]  # treating threshold as the pred

    print(mask.shape, pred.sum())
    mask_visualizer.plot_mask_predictions(patient, image, pred, gt)

    dd = dice_score(gt, pred).item()
    print('dice score', dd)
    scores[patient] = dd

print('all dice scores', scores)
with open(os.path.join(root_dir, 'dice_scores.txt'), 'w') as f:
    json.dump(scores, f, indent=4)