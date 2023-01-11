import json
import pickle
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from seg_3d.data.dataset import ImageToImage3D, JointTransform3D
from seg_3d.evaluation.mask_visualizer import MaskVisualizer


def dice_score(gt, pred):
    return (pred[gt == 1]).sum() * 2.0 / (pred.sum() + gt.sum())


# load dataset
dataset = ImageToImage3D(
    dataset_path='data/image_dataset',
    modality_roi_map=[{'CT': ['Inter', 'Threshold']}, {'PT': ['Bladder', 'Tumor', 'Tumor2', 'Tumor3']}],
    class_labels=['Background', 'Bladder', 'Tumor', 'Inter', 'Threshold'],
    slice_shape=(512, 512),
    joint_transform=JointTransform3D(test=True, crop=(200, 200))
)
print(dataset.patient_keys, '\n', len(dataset.patient_keys))

# plot_dir = os.path.join('seg_3d/output/ztoooo_delete', "masks")
# mask_visualizer = MaskVisualizer(class_labels=['Inter'], root_plot_dir=plot_dir,
                                    #  crop_size=(180, 180), save_figs=True)

dice_scores = {}
for i in range(59):
    p = dataset[i]
    print('\n', p['patient'])
    im, mask = p['image'], p['gt_mask']

    print(mask.shape, mask[-1].sum())
    # mask_visualizer.plot_mask_predictions(p['patient'],
                                            #   im, mask[-1].numpy(), mask[3].numpy(),
                                            #   skip_bkg=True, gt_overlay=False)

    
    dd = dice_score(mask[3], mask[-1])
    print('dice score', dd.item())
    dice_scores[p['patient']] = dd.item()
    print(dice_scores)

print('all dice scores', dice_scores)
