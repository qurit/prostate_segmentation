import json
import tqdm
import os
import pickle
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from collections import Counter
import skimage
from skimage import measure
import scipy

from bladder_segmentation.masks import SobelMask, CannyMask, MarchSquaresMask, EnsembleMeanMask, EnsembleMeanMaskPruned
from utils import contour2mask, centre_crop, dice_score
from dicom_code.contour_utils import parse_dicom_image

DATA_DIR = 'data'
CROP_SIZE = 100  # Size of the center crop on each frame to feed as input to the edge detection algorithms (reduces computation)


def generate_mask_predictions(*algorithms, dataset="image_dataset", show_mask=False, show_mask_algos=None,
                              show_hist=False, save_figs=False, slice_axes=None, multiview_alg="Ensemble-Mean",
                              bladder_frame_mode="pred", pred_dict=None, run_name=""):
    """
    Mask prediction step for the bladder segmentation pipeline. Will compute masks for all edge detection algorithms
    specified in the *algorithms arg.

    The following are the 2 modes specified by 'bladder_frame_mode' arg:
        1. "pred" - Run edge detection on the union of the predicted and ground truth bladder frame range, then zero-out frames
           not part of the predicted bladder frame range (we run on union to visualize the false positive and false
           negative frames)
        2. "gt" - Run edge detection on all frames part of the ground truth bladder frame range

    Args:
        *algorithms: Variable-length argument list for the specified edge detection algorithms
        dataset: Name of the dataset inside DATA_DIR, "image_dataset" or "test_dataset"
        slice_axes: The list of axes to perform slicing of frames, default is ["z"] when set to None
        multiview_alg: Algorithm used in the multiview step, i.e. ensemble across slice axes
        show_mask: Option to plot the predicted and ground truth masks for each frame in the bladder range
        show_mask_algos: List of algorithms to include in the mask plots, if None then show all
        show_hist: Option to show the histogram of dice scores across all scans
        save_figs: Option to save the outputted figures
        run_name: Name of the mask prediction run and directory name where script outputs are stored
        bladder_frame_mode: Specifies the mode "pred" (1), or "gt" (2) Default is "pred"
        pred_dict: Specifies subset of patients to run edge detection and predictions of bladder frame range
                   dict structure is the following: {"patient-key": [list of frame indices containing bladder"}

    Returns:

    """
    if not slice_axes:
        slice_axes = ["z"]
    check_axes = [a for a in slice_axes if a in ['x', 'y', 'z']]
    assert (len(check_axes) == len(slice_axes))

    assert (bladder_frame_mode in ["gt", "pred"])

    with open(os.path.join(DATA_DIR, dataset, 'global_dict.json')) as f:
        data_dict = json.load(f)

    # dir where figs, histograms, and dict results are saved
    mask_dir = os.path.join("bladder_segmentation", "experiments", run_name)
    os.makedirs(mask_dir, exist_ok=True)

    # initialize dict to store experiments results
    results_dict = {}

    # if pred_dict is set then run edge detection on those patients otherwise get patients from dataset
    patient_keys = pred_dict.keys() if pred_dict else data_dict.keys()

    # NOTE: SPECIFY PARTICULAR PATIENT KEYS HERE
    # patient_keys = []

    for patient in tqdm.tqdm(patient_keys):
        print("\n######################\nPatient ID", patient)
        scan = data_dict[patient]['PT']
        rois = scan['rois']
        bladder = rois['Bladder']

        if show_mask and save_figs:
            os.makedirs(os.path.join(mask_dir, "mask_predictions", patient), exist_ok=True)

        # get the ground truth first and last bladder frame indices
        bladder_frames_gt = np.asarray([frame for frame, contour in enumerate(bladder) if contour != []])
        check_continuous = lambda l: sorted(l) == list(range(min(l), max(l) + 1))
        assert check_continuous(bladder_frames_gt)

        if bladder_frame_mode == "gt":
            frame_range = bladder_frames_gt

        elif bladder_frame_mode == "pred":
            bladder_frames_preds = pred_dict[patient]
            # get the union of the predicted frame range and gt frame range
            frame_range = np.union1d(bladder_frames_preds, bladder_frames_gt).astype(int)
            print("bladder_frames_gt:    {}\n"
                  "bladder_frames_preds: {}\n"
                  "frame_range:          {}\n".format(bladder_frames_gt, bladder_frames_preds, frame_range))

        # get all frame file paths in bladder range
        frame_fps = [os.path.join(DATA_DIR, scan['fp'], str(frame) + '.dcm') for frame in frame_range]

        # generate 3d image from entire bladder frame range
        orig_img_3d = np.asarray([parse_dicom_image(dicom.dcmread(fp)) for fp in frame_fps])
        orig_img_size = np.shape(orig_img_3d)
        assert (len(orig_img_size) == 3)  # make sure image is 3d
        z_size, y_size, x_size = orig_img_size

        # generate the 3d ground truth mask
        ground_truth_3d = np.asarray([contour2mask(bladder[frame], orig_img_size[1:3]) for frame in frame_range])
        assert (np.shape(ground_truth_3d) == orig_img_size)

        # generate the 3d ground truth mask for all rois labeled 'Tumor'
        tumor_keys = [roi for roi in list(scan['rois'].keys()) if 'Tumor' in roi]
        tumor_list = [scan['rois'][tumor_key] for tumor_key in tumor_keys]
        tumor_mask_3d = np.zeros_like(orig_img_3d)
        for tumor in tumor_list:
            tumor_mask_3d += np.asarray([contour2mask(tumor[frame], orig_img_size[1:3]) for frame in frame_range])
        # set max value for pixel in mask to 1
        tumor_mask_3d[tumor_mask_3d > 1] = 1
        tumor_mask_3d = tumor_mask_3d.astype(int)
        assert (np.shape(tumor_mask_3d) == orig_img_size)

        # dict to store aggregated results
        results_dict[patient] = {
            "gt_mask_sum": ground_truth_3d.sum(), "gt_img_val_sums": orig_img_3d[ground_truth_3d == 1].sum(),
            "tumor_gt_overlap": ground_truth_3d[tumor_mask_3d == 1].sum(), "algos": {}
        }
        # dict to store masks for 1 patient
        patient_dict = {}

        # for each specified slice axis, run edge detection
        for curr_axis in slice_axes:
            if curr_axis == 'z':  # transverse plane
                # set depth axis to z - trivial case no need to transform 3d image
                trans_img = np.copy(orig_img_3d)

                # get centered crop to reduce computation
                crop_size = (CROP_SIZE, CROP_SIZE)
                # pass in a copied image object, otherwise orig_img gets modified
                img = centre_crop(np.copy(trans_img), (z_size, *crop_size))

                # apply initial thresholding
                img[img < 5000] = 0.

            elif curr_axis == 'y':  # frontal (coronal) plane
                # set depth axis to y - swap y and z
                trans_img = np.swapaxes(orig_img_3d, 1, 0)

                crop_size = (z_size, CROP_SIZE)
                img = centre_crop(np.copy(trans_img), (y_size, *crop_size))
                img[img < 5000] = 0.

            else:  # sagittal plane
                # set depth axis to x - swap x and z
                trans_img = np.swapaxes(orig_img_3d, 2, 0)

                crop_size = (CROP_SIZE, z_size)
                img = centre_crop(np.copy(trans_img), (x_size, *crop_size))
                img[img < 5000] = 0.

            trans_img_size = np.shape(trans_img)
            # iterate over each algorithm and compute the 3d mask
            for alg_fn in algorithms:
                alg_name = alg_fn().name
                curr_mask_3d = np.asarray([alg_fn.compute_mask(i) for i in img])

                full_mask_3d = np.zeros_like(trans_img)
                # find range of indices of center crop on full image size
                b1 = int(trans_img_size[1] / 2 + crop_size[0] / 2)
                a1 = b1 - crop_size[0]
                b2 = int(trans_img_size[2] / 2 + crop_size[1] / 2)
                a2 = b2 - crop_size[1]
                # apply the centered crop mask onto the full image size
                full_mask_3d[:, a1:b1, a2:b2] = curr_mask_3d

                # transform volume to original shape (z, y, x)
                if curr_axis == "z":
                    # trivial case
                    pass
                elif curr_axis == "y":
                    # swap z and y
                    full_mask_3d = np.swapaxes(full_mask_3d, 1, 0)
                else:
                    # swap z and x
                    full_mask_3d = np.swapaxes(full_mask_3d, 2, 0)

                if bladder_frame_mode == "pred":
                    # zero out frames which are not inside the predicted bladder frame range
                    for idx in frame_range:
                        if idx not in bladder_frames_preds:
                            full_mask_3d[idx - frame_range[0]] *= 0

                if alg_name == "Ensemble-Mean-Pruned":
                    labels = skimage.measure.label(full_mask_3d)
                    values, counts = np.unique(labels, return_counts=True)
                    # get most common label ignoring background
                    mode = values[1:][np.argmax(counts[1:])]
                    # set everything to 0 except for predicted bladder volume
                    labels[labels != mode] = 0
                    labels[labels == mode] = 1
                    full_mask_3d = labels

                    # do additional pruning
                    mask = []
                    for m in full_mask_3d:
                        m = scipy.ndimage.morphology.binary_fill_holes(m)
                        m = skimage.morphology.remove_small_objects(m.astype(bool), 15)
                        if np.sum(m) < 15:
                            m = np.zeros_like(m)
                        mask.append(m)
                    full_mask_3d = np.asarray(mask)

                # compute dice score for the whole volume
                # NOTE: remove this line to show ground truth overlap between bladder and tumor masks
                ground_truth_3d[tumor_mask_3d == 1] = 0
                dice = 0. if full_mask_3d.sum() == 0 else dice_score(ground_truth_3d, full_mask_3d)
                # compute the overlap between bladder mask and tumor ground truth mask
                tumor_pred_overlap = full_mask_3d[tumor_mask_3d == 1].sum()

                # add to results dict
                alg_name += "-" + curr_axis
                results_dict[patient]["algos"][alg_name] = {"dice": dice, "tumor_pred_overlap": tumor_pred_overlap,
                                                            "axis": curr_axis, "pred_mask_sum": full_mask_3d.sum(),
                                                            "pred_img_val_sums": orig_img_3d[full_mask_3d == 1].sum()}
                patient_dict[alg_name] = full_mask_3d

        if len(slice_axes) > 1 and multiview_alg:
            masks = []
            # get the masks for the specified alg
            for a in slice_axes:
                masks.append(patient_dict[multiview_alg + "-" + a])

            assert(len(np.shape(masks)) == 4)  # make sure this is a 4d array

            # do ensemble mean across views
            mv_mean_alg_name = "Mean MultiView-" + "".join(slice_axes)
            mv_mean_mask = (sum([m.astype(int) for m in masks]) + 0.00001) / len(masks)
            mv_mean_mask = np.round(mv_mean_mask).astype(int)

            # pruning
            mv_mean_mask = np.asarray([skimage.morphology.remove_small_objects(mask.astype(bool), 15) for mask in mv_mean_mask])

            # compute dice score for whole volume
            mv_mean_dice = 0. if mv_mean_mask.sum() == 0 else dice_score(ground_truth_3d, mv_mean_mask)
            # compute the overlap between bladder mask and tumor ground truth mask
            tumor_pred_overlap = mv_mean_mask[tumor_mask_3d == 1].sum()

            results_dict[patient]["algos"][mv_mean_alg_name] = {"dice": mv_mean_dice, "tumor_pred_overlap": tumor_pred_overlap,
                                                                "axis": slice_axes, "pred_mask_sum": mv_mean_mask.sum(),
                                                                "pred_img_val_sums": orig_img_3d[mv_mean_mask == 1].sum()}
            patient_dict[mv_mean_alg_name] = mv_mean_mask

        if show_mask:
            if show_mask_algos is None:
                show_mask_algos = patient_dict.keys()
            size = len(show_mask_algos) + 2
            crop_size = (CROP_SIZE, CROP_SIZE)

            # configure color maps
            raw_img_cmap = "inferno"
            gt_cmap = "OrRd"
            pred_cmap = "Greens"

            # configure legend for predictions
            cmap = {1: [*np.asarray([62, 33, 10])/255, 1.0],
                    2: [*np.asarray([129, 158, 131])/255, 1.0],
                    3: [*np.asarray([251, 249, 241])/255, 1.],
                    4: [*np.asarray([184, 126, 120]) / 255, 1.0]}
            labels = {1: 'TP', 2: 'FP', 3: 'TN', 4: 'FN'}
            pred_patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]

            # iterate through each slice
            for i, (orig_img_2d, gt_2d, tumor_2d) in enumerate(zip(orig_img_3d, ground_truth_3d, tumor_mask_3d)):
                # compute absolute index in scan
                abs_idx = i + frame_range[0]

                # check for any gt tumor and bladder overlap
                tumor_gt_overlap = gt_2d[tumor_2d == 1].sum()

                # configure legend for ground truth mask
                if tumor_gt_overlap != 0:
                    # case where there is bladder, tumor, and overlap
                    cmap = {1: list(cm.get_cmap(gt_cmap)(1.0)),
                            2: list(cm.get_cmap(gt_cmap)(0.33)),
                            3: list(cm.get_cmap(gt_cmap)(0.66))}
                    labels = {1: 'Bladder', 2: 'Tumor', 3: 'Overlap'}
                    # create patches as legend
                    gt_patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]

                elif tumor_2d.sum() != 0:
                    if gt_2d.sum() == 0:
                        # case where there is only tumor
                        a = int(gt_2d.shape[0] / 2 - CROP_SIZE / 2)
                        b = int(gt_2d.shape[1] / 2 - CROP_SIZE / 2)
                        tumor_2d[a][b] += 3  # hack to keep tumor color consistent, puts a bright pixel at (0,0) of the image so it is minimally visible
                        gt_patches = [mpatches.Patch(color=list(cm.get_cmap(gt_cmap)(0.33)), label='Tumor')]
                    else:
                        # case where there is both bladder and tumor
                        cmap = {1: list(cm.get_cmap(gt_cmap)(1.0)),
                                2: list(cm.get_cmap(gt_cmap)(0.33))}
                        labels = {1: 'Bladder', 2: 'Tumor'}
                        # create patches as legend
                        gt_patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]

                else:
                    # case where there is only bladder
                    # create patches as legend
                    gt_patches = [mpatches.Patch(color=list(cm.get_cmap(gt_cmap)(1.0)), label='Bladder')]

                # if only 1 alg make plot 1x3, otherwise make it two columns with as many rows required
                if len(show_mask_algos) == 1:
                    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
                    axs = np.expand_dims(axs, axis=0)
                else:
                    fig, axs = plt.subplots(int(np.ceil(size / 2)), 2, figsize=(10, max(10, round(len(show_mask_algos)/4 * 10))))
                axs[0, 0].imshow(orig_img_2d, cmap=raw_img_cmap)
                axs[0, 0].set_title('Original')
                axs[0, 0].set_xlabel("x")
                axs[0, 0].set_ylabel("y")
                axs[0, 1].imshow(centre_crop(np.abs(gt_2d - tumor_2d/3), crop_size), cmap=gt_cmap, interpolation='none', alpha=1.0)
                axs[0, 1].set_title('Ground Truth Mask\nGTSum: %.0f, B/ml: %.0f' % (gt_2d.sum(),
                                                                                    orig_img_2d[gt_2d == 1].sum()))
                axs[0, 1].legend(handles=gt_patches, loc=4)  # loc=1 for upper right

                dice_scores_title = ""
                for idx, alg_name in enumerate(show_mask_algos):
                    dice_scores_title += "%.04f " % (results_dict[patient]["algos"][alg_name]["dice"])
                    idx += 2

                    if len(show_mask_algos) == 1:
                        a, b = 0, 2
                    else:
                        # convert to 2d indices
                        a = int(idx / 2)
                        b = int(idx % 2)

                    pred_mask_2d = patient_dict[alg_name][i]
                    gt_2d[tumor_2d == 1] = 0
                    slice_dice = 0. if pred_mask_2d.sum() == 0 else dice_score(gt_2d, pred_mask_2d)
                    title = "%s Dice: %.04f\nPredSum: %.0f, B/ml: %.0f" % (alg_name, slice_dice, pred_mask_2d.sum(),
                                                                           orig_img_2d[pred_mask_2d == 1].sum())
                    tumor_pred_overlap = pred_mask_2d[tumor_2d == 1].sum()
                    if tumor_pred_overlap != 0:
                        title += "\nTumor Pred Overlap: %.0f" % tumor_pred_overlap

                    axs[a, b].imshow(centre_crop(gt_2d, crop_size), cmap=gt_cmap, interpolation='none', alpha=1.0)
                    axs[a, b].imshow(centre_crop(pred_mask_2d, crop_size), cmap=pred_cmap, interpolation='none', alpha=0.5)
                    axs[a, b].set_title(title)
                    axs[a, b].set_xticks([])
                    axs[a, b].set_yticks([])

                fig.suptitle("%s Order: %.0f, Dice Scores: %s" % (patient, abs_idx, dice_scores_title))
                # add a text indicator if classifier misclassified frame
                if abs_idx not in bladder_frames_gt:
                    # False Positive case
                    fig.text(0.5, 0.925, "(False Positive frame)", ha="center", va="bottom", size="medium", color="red")
                elif abs_idx not in bladder_frames_preds:
                    # False Negative case
                    fig.text(0.5, 0.925, "(False Negative frame)", ha="center", va="bottom", size="medium", color="red")

                # add legend for the prediction masks
                plt.legend(handles=pred_patches, loc=4)  # loc=1 for upper right
                # adjust spacing
                plt.tight_layout()
                plt.subplots_adjust(top=0.90)

                if save_figs:
                    # save fig to disk
                    fig.savefig(os.path.join(mask_dir, "mask_predictions", patient, str(abs_idx) + ".png"), format="png")
                    plt.close(fig)

                else:
                    plt.show()
                    plt.close('all')

    print("\nDone iterating through patients...\n")

    # save results to disk
    results_dict_fp = os.path.join(mask_dir, 'mask_prediction_results_dict' + "_" + bladder_frame_mode + '.pk')

    with open(results_dict_fp, 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved results to dict with file path: {}\n".format(results_dict_fp))

    # NOTE: CAN LOAD RESULTS_DICT HERE
    # results_dict = pickle.load(open(results_dict_fp, 'rb'), encoding='bytes')
    # NOTE: SPECIFY PARTICULAR ALGOS TO PLOT HERE
    # whitelist = []

    # aggregate dice scores for all edge detection algorithms
    algos_dict = {}
    for p in results_dict.keys():
        for a in results_dict[p]["algos"]:
            # if a in whitelist:
            if a not in algos_dict.keys():
                algos_dict[a] = {"dice": [results_dict[p]["algos"][a]["dice"]],
                                 "tumor_pred_overlap": [results_dict[p]["algos"][a]["tumor_pred_overlap"]]}
            else:
                algos_dict[a]["dice"].append(results_dict[p]["algos"][a]["dice"])
                algos_dict[a]["tumor_pred_overlap"].append(results_dict[p]["algos"][a]["tumor_pred_overlap"])

    # find the upper and bottom n patient scores and print them out
    upper_n = 3
    bottom_n = 3
    list_max_patients = []
    list_min_patients = []
    for algos_name in algos_dict.keys():
        # sort in increasing order
        scores = np.sort(algos_dict[algos_name]["dice"])
        max_scores = scores[-upper_n:]
        min_scores = scores[:bottom_n]

        # get the associated patient keys
        max_patients = list(filter(lambda x: results_dict[x]["algos"][algos_name]["dice"] in max_scores, results_dict))
        min_patients = list(filter(lambda x: results_dict[x]["algos"][algos_name]["dice"] in min_scores, results_dict))

        # keep track of all max and min patients
        list_max_patients.extend(max_patients)
        list_min_patients.extend(min_patients)

        # make dict
        dict_max_scores = {p: results_dict[p]["algos"][algos_name]["dice"] for p in max_patients}
        dict_min_scores = {p: results_dict[p]["algos"][algos_name]["dice"] for p in min_patients}

        # add to algos_dict
        algos_dict[algos_name]["max_scores"] = dict(sorted(dict_max_scores.items(), key=lambda item: item[1]))
        algos_dict[algos_name]["min_scores"] = dict(sorted(dict_min_scores.items(), key=lambda item: item[1]))

    # compute average scores across patients
    for alg_name in algos_dict.keys():
        print("{}\n\tmean dice score: {}\n\tmedian dice score: {}"
              "\n\tmax dice scores: {}\n\tmin dice scores: {}\n\tmean tumor pred overlap: {}"
              .format(alg_name,
                      np.mean(algos_dict[alg_name]["dice"]),
                      np.median(algos_dict[alg_name]["dice"]),
                      algos_dict[alg_name]["max_scores"],
                      algos_dict[alg_name]["min_scores"],
                      np.mean(algos_dict[alg_name]["tumor_pred_overlap"])))

    # print the set of max and min patients across all algos
    print("\nUpper {} patients which achieved max scores across all algos:\n\tset: {} \n\tcount: {}"
          .format(upper_n, list(Counter(list_max_patients).keys()), list(Counter(list_max_patients).values())))
    print("\nBottom {} patients which achieved min scores across all algos:\n\tset: {} \n\tcount: {}"
          .format(bottom_n, list(Counter(list_min_patients).keys()), list(Counter(list_min_patients).values())))

    # plots histograms to visualize results
    if show_hist:
        # set threshold below which lower values will be part of the same bin
        min_threshold = 0.5
        # plot hist of dice scores averaged across scans
        figsize = (6, round(len(algos_dict)/4 * 9))
        fig1, axs1 = plt.subplots(len(algos_dict.keys()), figsize=figsize, sharex=True)
        for idx, alg_name in enumerate(algos_dict.keys()):
            # put everything below min threshold in the same bin
            dist = np.asarray(algos_dict[alg_name]["dice"])
            dist_mean = np.mean(dist)
            dist[dist < min_threshold] = min_threshold

            axs1[idx].hist(dist)
            axs1[idx].set_title("%s\nMean: %.04f, Median: %.04f" % (alg_name, float(dist_mean), float(np.median(dist))))

            if idx == len(algos_dict.keys()) - 1:
                # generate x tick labels
                labels = list((np.asarray(axs1[idx].get_xticks()) * 100).astype(int) / 100)
                labels = ["< " + str(min_threshold) if label == min_threshold else label for label in labels]
                axs1[idx].set_xticklabels(labels)

        plt.suptitle("Histogram of Dice Scores Across Scans\nN = %.0f" % len(dist))
        plt.xlabel("Dice score")
        # adjust spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.80)

        if save_figs:
            fig1.savefig(os.path.join(mask_dir, "dice_scan_hist_" + bladder_frame_mode + ".png"), format="png")
        plt.show()
        plt.close('all')

        # plot hist of tumor overlap averaged across scans
        keys = ["Ground Truth", *algos_dict.keys()]  # TODO: want one of the bins to be 0
        fig2, axs2 = plt.subplots(len(keys), figsize=figsize, sharex=True)
        for idx, alg_name in enumerate(keys):
            # put ground truth at the top
            if idx == 0:
                dist = np.asarray([results_dict[p]["tumor_gt_overlap"] for p in results_dict.keys()])
            else:
                dist = np.asarray(algos_dict[alg_name]["tumor_pred_overlap"])
            dist_mean = np.mean(dist)

            axs2[idx].hist(dist)
            axs2[idx].set_title("%s\nMean Tumor Overlap: %.0f" % (alg_name, float(dist_mean)))

        plt.suptitle("Histogram of Tumor Overlap Across Scans\nN = %.0f" % len(dist))
        plt.xlabel("Pixel sum")
        # adjust spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        if save_figs:
            fig2.savefig(os.path.join(mask_dir, "tumor_overlap_scan_hist_" + bladder_frame_mode + ".png"), format="png")
        plt.show()
        plt.close('all')


if __name__ == '__main__':
    # get predictions from running inference
    model_dir = os.path.join("bladder_segmentation/experiments/models/run-cross-val-test-1/predictions_test_dataset.pk")
    preds = pickle.load(open(model_dir, 'rb'), encoding='bytes')
    preds = {p: np.where(preds[p] != 0)[0] for p in preds}
    print(preds)

    # code for loading up predictions on each val set from the 5-Fold CV
    # classifier_results = pickle.load(open("results_dict_pred_frame_finder_50x50.pk", 'rb'), encoding='bytes')
    #
    # specify classifier
    # clf = 'svc-rbf'
    # # get model predictions
    # model_dict = classifier_results['clf'][clf]
    #
    # # iterate through each fold and get the corresponding patients keys and predictions
    # preds = {}
    # for fold in model_dict.keys():
    #     for patient in model_dict[fold]['val']:
    #         arr = model_dict[fold]['val'][patient]['pred']
    #         arr = np.where(arr != 0)[0]
    #         preds[patient] = arr

    run_name = "test-dataset-edge-det-1"
    # add arbitrary number of edge detection algorithms in the args
    generate_mask_predictions(
                              EnsembleMeanMaskPruned,
                              dataset="test_dataset", slice_axes=['x', 'y', 'z'], run_name=run_name,
                              show_mask=True, show_hist=True, save_figs=True, bladder_frame_mode="pred", pred_dict=preds,
                              show_mask_algos=["Ensemble-Mean-Pruned-x"],
                              multiview_alg="Ensemble-Mean-Pruned"
                              )

# TODO:
# - need a better metric for tumor pred overlap, maybe just compute mean for patients that do have overlap
# - extract viz code
