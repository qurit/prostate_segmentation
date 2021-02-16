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


class SobelMask:
    def __init__(self):
        self.name = "Sobel"

    @staticmethod
    def compute_mask(img):
        mask = skimage.filters.sobel(img)
        markers = np.zeros_like(img)
        # set to background marker
        markers[img < 8000] = 1
        # set to bladder marker
        markers[img > np.max(img) * .15] = 2
        # separate bladder from background
        mask = skimage.segmentation.watershed(mask, markers)
        mask[mask == 1] = 0
        mask[mask == 2] = 1
        # remove objects smaller than min size
        mask = skimage.morphology.remove_small_objects(mask.astype(bool), min_size=30)

        canny = CannyMask.compute_mask(img)
        if mask.sum() < 5:
            mask = canny

        return mask


class CannyMask:
    def __init__(self):
        self.name = "Canny"

    @staticmethod
    def compute_mask(img):
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

        return mask


class MarchSquaresMask:
    def __init__(self):
        self.name = "Marching-Squares"

    @staticmethod
    def compute_mask(img, level=21000):
        contours = measure.find_contours(img, level=level)  # TODO: try mask param in find_contours

        if len(contours) == 0:
            mask = np.zeros(img.shape, int)

        else:
            # get the largest contour which should be the bladder
            list_len = [len(i) for i in contours]
            contours = [contours[np.argmax(list_len)]]
            # convert to array of ints
            contours = np.rint(contours).astype(int)
            # convert to mask
            mask = contour2mask(contours, img.shape, xy_ordering=False)  # TODO: why just an outline?? want to fill in hole

        return mask


class EnsembleMeanMask:
    def __init__(self):
        self.name = "Ensemble-Mean"

    @staticmethod
    def compute_mask(img):
        mask = [CannyMask.compute_mask(img),
                SobelMask.compute_mask(img),
                MarchSquaresMask.compute_mask(img)]

        mask = sum([m.astype(int) for m in mask]) / len(mask)
        mask = np.round(mask).astype(int)

        return mask


class EnsembleUnionMask:
    def __init__(self):
        self.name = "Ensemble-Union"

    @staticmethod
    def compute_mask(img):
        mask = [CannyMask.compute_mask(img),
                SobelMask.compute_mask(img),
                MarchSquaresMask.compute_mask(img)]

        mask = sum([m.astype(int) for m in mask])
        mask[mask > 1] = 1

        return mask


class DummyMask:
    def __init__(self):
        self.name = "Dummy"

    @staticmethod
    def compute_mask(img):
        return np.ones(img.shape, int)


# add more edge detection algorithms here


def mask_predictions(*algorithms, slice_axes=None, show_mask=False, show_hist=False, save_figs=False):
    if not slice_axes:
        slice_axes = ["z"]
    check_axes = [a for a in slice_axes if a in ['x', 'y', 'z']]
    assert (len(check_axes) == len(slice_axes))

    with open(os.path.join(DATA_DIR, 'test_dataset/global_dict.json')) as f:
        data_dict = json.load(f)

    # dir where figs are saved
    mask_dir = os.path.join(DATA_DIR, "mask_predictions")
    if save_figs:
        for axis in slice_axes:
            figs_dir = os.path.join(mask_dir, axis)
            if not os.path.isdir(figs_dir):
                os.makedirs(figs_dir)

    # initialize dict to store experiments results
    results_dict = {"algos": {}, "skipped": {}}

    for patient in tqdm.tqdm(data_dict.keys()):
        scan = data_dict[patient]['PT']
        rois = scan['rois']
        bladder = rois['Bladder']

        # dict to store dice score and mask results for 1 patient
        patient_pred_dict = {}

        # get the first and last bladder frame indices
        bladder_frames = [frame for frame, contour in enumerate(bladder) if contour != []]
        check_continuous = lambda l: sorted(l) == list(range(min(l), max(l) + 1))
        assert check_continuous(bladder_frames)

        # get all frame file paths in bladder range
        frame_fps = [os.path.join(DATA_DIR, scan['fp'], str(frame) + '.dcm') for frame in bladder_frames]

        # generate 3d image from entire bladder frame range
        orig_img_3d = np.asarray([parse_dicom_image(dicom.dcmread(fp)) for fp in frame_fps])
        orig_img_size = np.shape(orig_img_3d)
        assert (len(orig_img_size) == 3)  # make sure image is 3d
        z_size, y_size, x_size = orig_img_size

        # generate the 3d ground truth mask
        ground_truth_3d = np.asarray([contour2mask(bladder[frame], orig_img_size[1:3]) for frame in bladder_frames])
        assert (np.shape(ground_truth_3d) == orig_img_size)

        # skip frame if ground truth mask smaller than small threshold
        skip_threshold = 0
        # TODO: implement
        # if np.sum(ground_truth_3d) < skip_threshold:
        #     if patient not in results_dict['skipped'].keys():
        #         results_dict['skipped'][patient] = [i]
        #     else:
        #         results_dict['skipped'][patient].append(i)
        #     continue

        # for each specified slice axis, run edge detection
        for curr_axis in slice_axes:
            if curr_axis == 'z':
                # set principle axis to z - trivial case no need to transform 3d image or ground truth mask
                trans_img = np.copy(orig_img_3d)
                trans_gt = np.copy(ground_truth_3d)

                # get centered crop to reduce computation
                crop_size = (100, 100)
                # pass in a copied image object, otherwise orig_img gets modified
                img = centre_crop(np.copy(trans_img), (z_size, *crop_size))

                # apply initial thresholding
                img[img < 5000] = 0.

            elif curr_axis == 'y':
                # set principle axis to y - swap y and z
                trans_img = np.swapaxes(orig_img_3d, 1, 0)
                trans_gt = np.swapaxes(ground_truth_3d, 1, 0)

                crop_size = (z_size, 100)
                img = centre_crop(np.copy(trans_img), (y_size, *crop_size))
                img[img < 5000] = 0.  # TODO: find right value

            else:
                # TODO: find which direction this is anatomically
                # set principle axis to x - swap x and z
                trans_img = np.swapaxes(orig_img_3d, 2, 0)
                trans_gt = np.swapaxes(ground_truth_3d, 2, 0)

                crop_size = (100, z_size)
                img = centre_crop(np.copy(trans_img), (x_size, *crop_size))  # TODO: still use 100x100 center crop
                img[img < 5000] = 0.

            trans_img_size = np.shape(trans_img)
            # iterate over each algorithm and compute the 3d mask
            for alg_fn in algorithms:
                alg_name = alg_fn.__name__
                curr_mask_3d = [alg_fn.compute_mask(i) for i in img]
                # plt.imshow(curr_mask_3d[90])
                # plt.show()

                full_mask_3d = np.zeros_like(trans_img)
                # find range of indices of center crop on full image size
                b1 = int(trans_img_size[1] / 2 + crop_size[0] / 2)
                a1 = b1 - crop_size[0]
                b2 = int(trans_img_size[2] / 2 + crop_size[1] / 2)
                a2 = b2 - crop_size[1]
                # apply the centered crop mask onto the full image size
                full_mask_3d[:, a1:b1, a2:b2] = curr_mask_3d

                # compute dice score for the whole volume
                dice = 0. if full_mask_3d.sum() == 0 else dice_score(trans_gt, full_mask_3d)

                # print(alg_name, dice)

                # add to results dict
                to_save = {"dice": dice, "mask": full_mask_3d}
                if alg_name not in patient_pred_dict.keys():
                    patient_pred_dict[alg_name] = {curr_axis: to_save}
                else:
                    patient_pred_dict[alg_name][curr_axis] = to_save

            # TODO if show mask
            # - make new dir for ax
            # - save out 3d pred mask and gt
            # - save out each frame and find the dice score
            if show_mask:
                # TODO plot 3d gt and pred mask

                size = len(patient_pred_dict.keys()) + 2

                # iterate through each slice
                # TODO: only save out slices that contain bladder
                for i, (slice_2d, gt_2d) in enumerate(zip(trans_img, trans_gt)):

                    fig, axs = plt.subplots(int(np.ceil(size / 2)), 2, figsize=(10, 10))
                    axs[0, 0].imshow(slice_2d)
                    axs[0, 0].set_title('Original')
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    axs[0, 1].imshow(gt_2d, cmap='Dark2', interpolation='none', alpha=0.7)
                    axs[0, 1].set_title('Ground Truth Mask')
                    axs[0, 1].set_xticks([])
                    axs[0, 1].set_yticks([])

                    for idx, alg_name in enumerate(patient_pred_dict.keys()):
                        idx += 2
                        # convert to 2d indices
                        a = int(idx / 2)
                        b = int(idx % 2)

                        slice_mask = patient_pred_dict[alg_name][curr_axis]['mask'][i]
                        axs[a, b].imshow(slice_mask, cmap='gray', interpolation='none')
                        axs[a, b].imshow(gt_2d, cmap='Dark2', interpolation='none', alpha=0.7)
                        slice_dice = 0. if slice_mask.sum() == 0 else dice_score(gt_2d, slice_mask)
                        axs[a, b].set_title("%s Dice Score %.04f" % (alg_name, slice_dice))
                        axs[a, b].set_xticks([])
                        axs[a, b].set_yticks([])

                    fig.suptitle("Patient ID: %s\nPrinciple axis: %s" % (patient, curr_axis))
                    # adjust spacing
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.90)

                    if save_figs:
                        # save fig to disk
                        fig.savefig(os.path.join(mask_dir, curr_axis, patient + "-" + str(i) + ".png"), format="png")
                        plt.close(fig)

                    else:
                        plt.show()
                        plt.close('all')

        # add to results dict
        for alg in patient_pred_dict.keys():
            a = patient_pred_dict[alg]
            for axis in a.keys():
                key_name = alg + "-" + axis
                if key_name not in results_dict['algos'].keys():
                    results_dict['algos'][key_name] = {"dice": [a[axis]["dice"]], "axis": axis}
                else:
                    results_dict['algos'][key_name]['dice'].append(a[axis]['dice'])

        # TODO: do multi-view ensembling here, then save to results_dict

    # compute + print skipped frame stats
    flatten = lambda t: [item for sublist in t for item in sublist]
    all_skipped = flatten([*results_dict['skipped'].values()])
    average_skipped = [len(val) for val in results_dict['skipped'].values()]
    print("Skipped %.0f frames when sum(ground_truth) < %.0f" % (len(all_skipped), skip_threshold))
    print("Average frames skipped per patient %.2f" % (sum(average_skipped) / len(data_dict.keys())))
    print("Skipped frames object\n", results_dict['skipped'], "\n")

    # compute average scores across patients
    algos_dict = results_dict['algos']
    for alg_name in algos_dict.keys():
        avg = np.mean(algos_dict[alg_name]["dice"])
        print("{} average dice score: {}".format(alg_name, avg))

    # plots histograms of the dice scores for each algorithm
    if show_hist:
        # set threshold below which lower values will be part of the same bin
        min_threshold = 0.7
        # plot hist of dice scores averaged across scans
        figsize = (6, int(len(algos_dict)/6 * 9))
        fig1, axs1 = plt.subplots(len(algos_dict.keys()), figsize=figsize, sharex=True)
        for idx, alg_name in enumerate(algos_dict.keys()):
            # put everything below min threshold in the same bin
            dist = np.asarray(algos_dict[alg_name]["dice"])
            dist_mean = np.mean(dist)
            dist[dist < min_threshold] = min_threshold

            axs1[idx].hist(dist)
            axs1[idx].set_title("%s Mean Dice Score %.04f" % (alg_name, float(dist_mean)))

            if idx == len(algos_dict.keys()) - 1:
                # generate x tick labels
                labels = list((np.asarray(axs1[idx].get_xticks()) * 100).astype(int) / 100)
                labels = ["< " + str(min_threshold) if label == min_threshold else label for label in labels]
                axs1[idx].set_xticklabels(labels)

        plt.suptitle("Histogram of Dice Scores Averaged Across Scans\nN = %.0f" % len(dist))
        # adjust spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        if save_figs:
            fig1.savefig(os.path.join(mask_dir, "dice_scan_hist.png"), format="png")
        plt.show()
        plt.close('all')

        # TODO: plot sums across each slice and compare to ground truth


if __name__ == '__main__':
    # add arbitrary number of edge detection algorithms in the args
    mask_predictions(CannyMask,
                     SobelMask,
                     MarchSquaresMask,
                     EnsembleMeanMask, slice_axes=['x', 'y', 'z'], show_mask=True, show_hist=False, save_figs=True)

# experiments to try
# - remove all small objects
# - play around with marching squares
# - special logic for first and last bladder frame
# - sobel, canny params, add prewitt edge detection
# - pipeline approaches
# - optimize params of the current algos
# - what if we dont find the bladder range, and just use the seed and the algorithm finds the edges
# - or bladder finder just knows the partial bladder range

# - 3d

# by wed morn
# - 2d + frame classifier, pass in 0-100 frames
# - multi-view 2d, compare all 3 views, ensemble


# - end-to-end ml pipeline

# - handcrafted, random forest, image classification
# - multi-view 2d, compare all 3 views, ensemble


# NEXT setup multi-view 2d
