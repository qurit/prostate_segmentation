import json
import tqdm
import scipy
import os
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt

import skimage
import skimage.filters
import skimage.feature
import skimage.segmentation
from skimage import measure
from utils import contour2mask, centre_crop, dice_score
from dicom_code.contour_utils import parse_dicom_image


DATA_DIR = 'data'


def sobel_mask(img, alg_name="Sobel"):
    mask = skimage.filters.sobel(img)
    markers = np.zeros_like(img)
    markers[img < 8000] = 1
    markers[img > np.max(img) * .15] = 2
    mask = skimage.segmentation.watershed(mask, markers)
    mask[mask == 1] = 0
    mask[mask == 2] = 1
    mask = skimage.morphology.remove_small_objects(mask.astype(bool), 30)

    if mask.sum() > 100:
        mask = skimage.morphology.remove_small_objects(mask.astype(bool), 30)
    else:
        mask = skimage.morphology.remove_small_objects(mask.astype(bool), 5)

    _, canny = canny_mask(img)
    if mask.sum() < 5:
        mask = canny

    return alg_name, mask


def canny_mask(img, alg_name="Canny"):
    mask = skimage.feature.canny(img, low_threshold=5000, sigma=4.2)
    markers = np.zeros_like(img)
    markers[img < 8000] = 1
    markers[img > np.max(img) * .25] = 2
    mask = skimage.segmentation.watershed(mask, markers)
    mask[mask == 1] = 0
    mask[mask == 2] = 1
    mask = scipy.ndimage.morphology.binary_fill_holes(mask)

    if mask.sum() > 100:
        mask = skimage.morphology.remove_small_objects(mask.astype(bool), 30)
    else:
        mask = skimage.morphology.remove_small_objects(mask.astype(bool), 25)

    return alg_name, mask


def march_squares_mask(img, alg_name="Marching Squares"):
    contours = measure.find_contours(img, level=20000)

    if len(contours) == 0:
        mask = np.zeros(img.shape, int)

    else:
        # get the largest contour which should be the bladder
        list_len = [len(i) for i in contours]
        contours = [contours[np.argmax(list_len)]]
        # convert to array of ints
        contours = np.rint(contours).astype(int)
        # convert to mask
        mask = contour2mask(contours, img.shape, xy_ordering=False)

    return alg_name, mask


def dummy_mask(img, alg_name="Dummy"):
    return alg_name, np.ones(img.shape, int)


# add more edge detection algorithms here


def mask_predictions(*algorithms, show_mask=False):
    with open(os.path.join(DATA_DIR, 'image_dataset/global_dict.json')) as f:
        data_dict = json.load(f)

    # initialize dict to store dice score results
    global_dice_dict = {}

    for patient in tqdm.tqdm(data_dict.keys()):
        scan = data_dict[patient]['PT']
        rois = scan['rois']
        bladder = rois['Bladder']

        # get the first and last bladder frame indices
        bladder_frames = [frame for frame, contour in enumerate(bladder) if contour != []]
        bladder_frames = (bladder_frames[0], bladder_frames[-1])

        # dict to store dice score and mask results for 1 patient
        patient_pred_dict = {}

        for i, frame in enumerate(range(*bladder_frames)):
            # load dicom file
            img_dcm = dicom.dcmread(os.path.join(DATA_DIR, scan['fp'], str(frame) + '.dcm'))
            orig_img = parse_dicom_image(img_dcm)

            # reduces computation
            img = centre_crop(np.copy(orig_img), (100, 100))  # pass in a copied image object, otherwise orig_img gets modified

            # apply initial thresholding
            img[img < 5000] = 0.

            # get ground truth mask
            ground_truth = contour2mask(bladder[frame], orig_img.shape)

            for alg in algorithms:
                alg_name, curr_mask = alg(img)
                full_mask = np.zeros_like(orig_img)
                full_mask[46:146, 46:146] = curr_mask

                dice = 0. if full_mask.sum() == 0 else dice_score(ground_truth, full_mask)

                # add to patient dict
                if alg_name not in patient_pred_dict.keys():
                    patient_pred_dict[alg_name] = {}
                    patient_pred_dict[alg_name]["dice"] = [dice]
                    patient_pred_dict[alg_name]["mask"] = [full_mask]
                else:
                    patient_pred_dict[alg_name]["dice"].append(dice)
                    patient_pred_dict[alg_name]["mask"].append(full_mask)

            if show_mask:
                size = len(patient_pred_dict.keys()) + 2
                fig, axs = plt.subplots(int(np.ceil(size/2)), 2, figsize=(10, 10))
                axs[0, 0].imshow(orig_img)
                axs[0, 0].set_title('Original')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(ground_truth, cmap='Dark2', interpolation='none', alpha=0.7)
                axs[0, 1].set_title('Ground Truth Mask')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])

                for idx, alg_name in enumerate(patient_pred_dict.keys()):
                    idx += 2
                    # convert to 2d indices
                    a = int(idx / 2)
                    b = int(idx % 2)

                    axs[a, b].imshow(patient_pred_dict[alg_name]['mask'][i], cmap='gray', interpolation='none')
                    axs[a, b].imshow(ground_truth, cmap='Dark2', interpolation='none', alpha=0.7)
                    axs[a, b].set_title("%s Prediction Dice Score %.04f" % (alg_name, patient_pred_dict[alg_name]['dice'][i]))  # + ' ' + str(round(np.max(img)/10000, 5)))
                    axs[a, b].set_xticks([])
                    axs[a, b].set_yticks([])

                fig.suptitle("Patient ID: %s" % patient)
                # plt.subplots_adjust(top=0.85)  # TODO: add spacing between sup title and subplots
                plt.show()
                plt.close('all')

        # add to global dice dict
        for alg in patient_pred_dict.keys():
            # compute average dice across scan
            if alg not in global_dice_dict.keys():
                global_dice_dict[alg] = [np.mean(patient_pred_dict[alg]["dice"])]
            else:
                global_dice_dict[alg].append(np.mean(patient_pred_dict[alg]["dice"]))

    # compute average scores across patients
    for alg in global_dice_dict.keys():
        avg = np.mean(global_dice_dict[alg])
        print("{} dice score: {}".format(alg, avg))


if __name__ == '__main__':
    # add arbitrary number of edge detection algorithms in the args
    mask_predictions(sobel_mask, canny_mask, march_squares_mask, dummy_mask, show_mask=False)
