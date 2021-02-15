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


def march_squares_mask(img, alg_name="Marching-Squares"):
    level = 15000
    contours = measure.find_contours(img, level=level)  # TODO: try mask param in find_contours
    # print(level/np.mean(img))

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

    _, canny = canny_mask(img)
    if mask.sum() < 5:
        mask = canny

    return alg_name, mask


def ensemble_mean(img, alg_name="Ensemble-Mean"):
    _, canny = canny_mask(img)
    _, sobel = sobel_mask(img)
    _, march = march_squares_mask(img)
    mask = (canny.astype(int) + sobel.astype(int) + march.astype(int)) / 3
    mask = np.round(mask).astype(int)

    return alg_name, mask


def ensemble_union(img, alg_name="Ensemble-Union"):
    _, canny = canny_mask(img)
    _, sobel = sobel_mask(img)
    _, march = march_squares_mask(img)
    mask = (canny.astype(int) + sobel.astype(int) + march.astype(int))
    mask[mask > 1] = 1

    return alg_name, mask


def dummy_mask(img, alg_name="Dummy"):
    return alg_name, np.ones(img.shape, int)


# add more edge detection algorithms here


def mask_predictions(*algorithms, show_mask=False, show_hist=False):
    with open(os.path.join(DATA_DIR, 'image_dataset/global_dict.json')) as f:
        data_dict = json.load(f)

    # initialize dict to store dice score results
    global_dice_dict = {}

    # initialize dict to store skipped frames
    skipped_frames = {}

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
            orig_img_size = np.shape(orig_img)

            # get ground truth mask
            ground_truth = contour2mask(bladder[frame], orig_img_size)

            # skip frame if ground truth mask smaller than small threshold
            skip_threshold = 0
            if np.sum(ground_truth) < skip_threshold:
                if patient not in skipped_frames.keys():
                    skipped_frames[patient] = [i]
                else:
                    skipped_frames[patient].append(i)
                continue

            # reduces computation
            crop_size = 100
            img = centre_crop(np.copy(orig_img), (crop_size, crop_size))  # pass in a copied image object, otherwise orig_img gets modified

            # apply initial thresholding
            img[img < 5000] = 0.

            for alg_name in algorithms:
                alg_name, curr_mask = alg_name(img)
                full_mask = np.zeros_like(orig_img)
                # find range of indices of center crop on original image
                b = int(orig_img_size[0]/2 + crop_size/2)
                a = b - crop_size
                full_mask[a:b, a:b] = curr_mask

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
                # adjust spacing
                plt.tight_layout()
                plt.subplots_adjust(top=0.90)

                # save fig to disk
                mask_dir = os.path.join(DATA_DIR, "mask_predictions")
                if not os.path.isdir(mask_dir):
                    os.makedirs(mask_dir)
                fig.savefig(os.path.join(mask_dir, patient + "-" + str(i) + ".png"), format="png")
                plt.close(fig)

                # plt.show()
                # plt.close('all')

        # add to global dice dict
        for alg_name in patient_pred_dict.keys():
            # compute average dice across scan
            if alg_name not in global_dice_dict.keys():
                global_dice_dict[alg_name] = [np.mean(patient_pred_dict[alg_name]["dice"])]
            else:
                global_dice_dict[alg_name].append(np.mean(patient_pred_dict[alg_name]["dice"]))

    # compute + print skipped frame stats
    flatten = lambda t: [item for sublist in t for item in sublist]
    all_skipped = flatten([*skipped_frames.values()])
    average_skipped = [len(val) for val in skipped_frames.values()]
    print("Skipped %.0f frames when sum(ground_truth) < %.0f" % (len(all_skipped), skip_threshold))
    print("Average frames skipped per patient %.2f" % (sum(average_skipped)/len(data_dict.keys())))
    print("Skipped frames object\n", skipped_frames, "\n")

    # compute average scores across patients
    for alg_name in global_dice_dict.keys():
        avg = np.mean(global_dice_dict[alg_name])
        print("{} dice score: {}".format(alg_name, avg))

    # plots histograms of the dice scores for each algorithm
    if show_hist:
        fig, axs = plt.subplots(len(global_dice_dict.keys()), figsize=(6, 9), sharex=True)
        for idx, alg_name in enumerate(global_dice_dict.keys()):
            axs[idx].hist(global_dice_dict[alg_name])
            axs[idx].set_title("%s Mean Dice Score %.04f" % (alg_name, float(np.mean(global_dice_dict[alg_name]))))

        plt.suptitle("Histogram of Dice Scores")
        # adjust spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()
        plt.close('all')


if __name__ == '__main__':
    # add arbitrary number of edge detection algorithms in the args
    mask_predictions(canny_mask,
                     sobel_mask,
                     march_squares_mask,
                     ensemble_mean,
                     ensemble_union, show_mask=False, show_hist=False)
