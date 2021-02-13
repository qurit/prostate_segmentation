import json
import tqdm
import scipy
import os
import numpy as np
import pydicom as dicom

import skimage.filters
import skimage.segmentation
from skimage import feature
from utils import contour2mask, centre_crop, dice_score
from dicom_code.contour_utils import parse_dicom_image

root = '/home/yous/Desktop/ryt/'
sobel_global, canny_global, all_frames = [], [], []

with open('/home/yous/Desktop/ryt/image_dataset/global_dict.json') as f:
    data_dict = json.load(f)

for patient in tqdm.tqdm(data_dict.keys()):
    scan = data_dict[patient]['PT']
    rois = scan['rois']
    bladder = rois['Bladder']

    search_range = (int(0.05*len(bladder)), int(.65*len(bladder)))
    bladder_frames = [frame for frame, contour in enumerate(bladder) if contour != []]
    bladder_frames = (bladder_frames[0], bladder_frames[-1])

    pat_sob_dcs, pat_can_dcs = [], []

    for frame in range(*bladder_frames):
        img_dcm = dicom.dcmread(os.path.join(root, scan['fp'], str(frame)+'.dcm'))
        orig_img = parse_dicom_image(img_dcm)
        img = centre_crop(orig_img, (100, 100))

        img[img < 5000] = 0.
        sobel_pred = skimage.filters.sobel(img)
        markers = np.zeros_like(img)
        markers[img < 8000] = 1
        markers[img > np.max(img) * .15] = 2
        sobel_pred = skimage.segmentation.watershed(sobel_pred, markers)
        sobel_pred[sobel_pred == 1] = 0
        sobel_pred[sobel_pred == 2] = 1
        sobel_pred = skimage.morphology.remove_small_objects(sobel_pred.astype(bool), 30)

        canny_pred = feature.canny(img, low_threshold=5000, sigma=4.2)
        markers = np.zeros_like(img)
        markers[img < 8000] = 1
        markers[img > np.max(img) * .25] = 2
        canny_pred = skimage.segmentation.watershed(canny_pred, markers)
        canny_pred[canny_pred == 1] = 0
        canny_pred[canny_pred == 2] = 1
        canny_pred = scipy.ndimage.morphology.binary_fill_holes(canny_pred)

        if sobel_pred.sum() > 100:
            sobel_pred = skimage.morphology.remove_small_objects(sobel_pred.astype(bool), 30)
        else:
            sobel_pred = skimage.morphology.remove_small_objects(sobel_pred.astype(bool), 5)

        if canny_pred.sum() > 100:
            canny_pred = skimage.morphology.remove_small_objects(canny_pred.astype(bool), 30)
        else:
            canny_pred = skimage.morphology.remove_small_objects(canny_pred.astype(bool), 25)

        if sobel_pred.sum() < 5:
            sobel_pred = canny_pred

        sobel_mask, canny_mask = np.zeros_like(orig_img), np.zeros_like(orig_img)
        sobel_mask[46:146, 46:146] = sobel_pred
        canny_mask[46:146, 46:146] = canny_pred

        ground_truth = contour2mask(bladder[frame], orig_img.shape)

        if sobel_mask.sum() == 0:
            sobel_dice = 0.
        else:
            sobel_dice = dice_score(ground_truth, sobel_mask)

        if canny_mask.sum() == 0:
            canny_dice = 0.
        else:
            canny_dice = dice_score(ground_truth, canny_mask)

        pat_sob_dcs.append(sobel_dice), pat_can_dcs.append(canny_dice)

        # if canny_pred.sum() < 150:
        #     fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 10))
        #     fig.suptitle(patient)
        #     ax1[0].imshow(orig_img)
        #     ax1[0].set_title('Original ')
        #     ax1[1].imshow(ground_truth, cmap='Dark2', interpolation='none', alpha=0.7)
        #     ax1[1].set_title('Ground Truth Mask')
        #     ax2[0].imshow(sobel_mask)
        #     ax2[0].imshow(ground_truth, cmap='Dark2', interpolation='none', alpha=0.7)
        #     ax2[0].set_title('Sobel Prediction ' + str(sobel_dice) + ' ' + str(round(np.max(img)/10000, 5)))
        #     ax2[1].imshow(canny_mask, cmap='gray', interpolation='none')
        #     ax2[1].imshow(ground_truth, cmap='Dark2', interpolation='none', alpha=0.7)
        #     ax2[1].set_title('Canny Prediction ' + str(round(canny_dice, 3))+' '+str(canny_pred.sum()))
        #     plt.show()
        #     plt.close('all')

    all_frames.extend(pat_sob_dcs)
    sobel_avg_score = sum(pat_sob_dcs) / len(pat_sob_dcs)
    sobel_global.append(sobel_avg_score)

    canny_avg_score = sum(pat_can_dcs) / len(pat_can_dcs)
    canny_global.append(canny_avg_score)

print(sum(sobel_global) / len(sobel_global))
print(sum(canny_global) / len(canny_global))

