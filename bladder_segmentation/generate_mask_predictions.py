import json
import tqdm
import os
import pickle
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from bladder_segmentation.masks import SobelMask, CannyMask, MarchSquaresMask, EnsembleMeanMask
from utils import contour2mask, centre_crop, dice_score
from dicom_code.contour_utils import parse_dicom_image

DATA_DIR = 'data'
FRAMES_PER_PATIENT = 100  # TODO: make adaptive
CROP_SIZE = 100


def generate_mask_predictions(*algorithms, dataset="image_dataset", show_mask=False, show_mask_algos=None,
                              show_hist=False, save_figs=False, slice_axes=None, multiview_alg="Ensemble-Mean",
                              bladder_frame_mode="gt", pred_dict=None, run_name=""):
    """
    Mask prediction step for the bladder segmentation pipeline. Will compute masks for all edge detection algorithms
    specified in the *algorithms arg.

    The following are the 3 modes specified by 'bladder_frame_mode' arg:
        1. Run edge detection on all frames part of the specified frame range 0,...,FRAMES_PER_PATIENT
        2. Run edge detection on all frames part of the ground truth bladder frame range
        3. Run edge detection on all frames, then zero-out frames not part of the predicted bladder frame range

    Args:
        *algorithms: Variable-length argument list for the specified edge detection algorithms
        dataset: Name of the dataset inside DATA_DIR, "image_dataset" or "test_dataset"
        slice_axes: The list of axes to perform slicing of frames, default is ["z"] when set to None
        multiview_alg: Algorithm used in the multiview step, i.e. ensemble across slice axes
        show_mask: Option to plot the predicted and ground truth masks for each frame in the bladder range
        show_mask_algos: List of algorithms to include in the mask plots, if None then show all
        show_hist: Option to show the histogram of dice scores across all scans
        save_figs: Option to save the outputted figures
        run_name: Option to give a name to the mask prediction run which will add a suffix to the saved dice hist plot
        bladder_frame_mode: Specifies the mode "all" (1), "gt" (2), or "pred" (3) Default is "gt"
        pred_dict: Specifies subset of patients to run edge detection and predictions of bladder frame range
                   dict structure is the following: {"patient-key": [list of frame indices containing bladder"}

    Returns:

    """
    if not slice_axes:
        slice_axes = ["z"]
    check_axes = [a for a in slice_axes if a in ['x', 'y', 'z']]
    assert (len(check_axes) == len(slice_axes))
    axis_label_map = {"z": ["x", "y"], "y": ["x", "z"], "x": ["z", "y"]}

    assert (bladder_frame_mode in ["all", "gt", "pred"])

    with open(os.path.join(DATA_DIR, dataset, 'global_dict.json')) as f:
        data_dict = json.load(f)

    # dir where figs are saved
    mask_dir = os.path.join(DATA_DIR, "mask_predictions")
    if save_figs:
        for axis in slice_axes:
            figs_dir = os.path.join(mask_dir, axis)
            if not os.path.isdir(figs_dir):
                os.makedirs(figs_dir)

    # initialize dict to store experiments results
    results_dict = {}

    # specify which frames to consider from scan
    if bladder_frame_mode in ["all", "pred"]:
        frame_range = np.arange(0, FRAMES_PER_PATIENT)

    # if pred_dict is set then run edge detection on those patients otherwise get patients from dataset
    patient_keys = pred_dict.keys() if pred_dict else data_dict.keys()

    for patient in tqdm.tqdm(patient_keys):
        scan = data_dict[patient]['PT']
        rois = scan['rois']
        bladder = rois['Bladder']

        if bladder_frame_mode == "gt":
            # get the ground truth first and last bladder frame indices
            bladder_frames_gt = [frame for frame, contour in enumerate(bladder) if contour != []]
            check_continuous = lambda l: sorted(l) == list(range(min(l), max(l) + 1))
            assert check_continuous(bladder_frames_gt)
            frame_range = bladder_frames_gt

        # get the predicted bladder frame range
        if bladder_frame_mode == "pred":
            bladder_frames_preds = pred_dict[patient]

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
        results_dict[patient] = {"gt_sum": ground_truth_3d.sum(), "algos": {}}
        # dict to store masks for 1 patient
        patient_dict = {}

        # for each specified slice axis, run edge detection
        for curr_axis in slice_axes:
            if curr_axis == 'z':
                # set depth axis to z - trivial case no need to transform 3d image
                trans_img = np.copy(orig_img_3d)

                # get centered crop to reduce computation
                crop_size = (CROP_SIZE, CROP_SIZE)
                # pass in a copied image object, otherwise orig_img gets modified
                img = centre_crop(np.copy(trans_img), (z_size, *crop_size))

                # apply initial thresholding
                img[img < 5000] = 0.

            elif curr_axis == 'y':
                # set depth axis to y - swap y and z
                trans_img = np.swapaxes(orig_img_3d, 1, 0)

                crop_size = (z_size, CROP_SIZE)
                img = centre_crop(np.copy(trans_img), (y_size, *crop_size))
                img[img < 5000] = 0.

            else:
                # set depth axis to x - swap x and z
                trans_img = np.swapaxes(orig_img_3d, 2, 0)

                crop_size = (CROP_SIZE, z_size)
                img = centre_crop(np.copy(trans_img), (x_size, *crop_size))
                img[img < 5000] = 0.

            trans_img_size = np.shape(trans_img)
            # iterate over each algorithm and compute the 3d mask
            for alg_fn in algorithms:
                alg_name = alg_fn().name
                curr_mask_3d = [alg_fn.compute_mask(i) for i in img]

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
                            full_mask_3d[idx] *= 0

                # compute dice score for the whole volume
                dice = 0. if full_mask_3d.sum() == 0 else dice_score(ground_truth_3d, full_mask_3d)
                # compute the overlap between bladder mask and tumor ground truth mask
                tumor_pred_overlap = full_mask_3d[tumor_mask_3d == 1].sum()

                # add to results dict
                alg_name += "-" + curr_axis
                results_dict[patient]["algos"][alg_name] = {"dice": dice, "tumor_pred_overlap": tumor_pred_overlap,
                                                            "axis": curr_axis, "pred_sum": full_mask_3d.sum()}
                patient_dict[alg_name] = full_mask_3d

        if len(slice_axes) > 1 and multiview_alg:
            # TODO: try intersection
            masks = []
            # get the masks for the specified alg
            for a in slice_axes:
                masks.append(patient_dict[multiview_alg + "-" + a])

            assert(len(np.shape(masks)) == 4)  # make sure this is a 4d array

            # do ensemble mean across views
            mv_mean_alg_name = "Mean MultiView-" + "".join(slice_axes)
            mv_mean_mask = (sum([m.astype(int) for m in masks]) + 0.00001) / len(masks)
            mv_mean_mask = np.round(mv_mean_mask).astype(int)

            # compute dice score for whole volume
            mv_mean_dice = 0. if mv_mean_mask.sum() == 0 else dice_score(ground_truth_3d, mv_mean_mask)
            # compute the overlap between bladder mask and tumor ground truth mask
            tumor_pred_overlap = mv_mean_mask[tumor_mask_3d == 1].sum()

            results_dict[patient]["algos"][mv_mean_alg_name] = {"dice": mv_mean_dice, "tumor_pred_overlap": tumor_pred_overlap,
                                                                "axis": slice_axes, "pred_sum": mv_mean_mask.sum()}

        if show_mask:  # and tumor_mask_3d.sum() != 0:
            # TODO: add image value sums
            if show_mask_algos is None:
                show_mask_algos = patient_dict.keys()
            size = len(show_mask_algos) + 2
            empty_mask_frames = [i for i, mask in enumerate(ground_truth_3d) if mask.sum() == 0]
            fig_num = 0
            x_label, y_label = axis_label_map["z"]
            crop_size = (50, 50)

            # configure color maps
            raw_img_cmap = "inferno"
            gt_cmap = "OrRd"
            pred_cmap = "Greens"

            # configure legend for ground truth mask
            cmap = {1: list(cm.get_cmap(gt_cmap)(1.0)),
                    2: list(cm.get_cmap(gt_cmap)(0.5))}
            labels = {1: 'Bladder', 2: 'Tumor'}
            # create patches as legend
            gt_patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]

            # configure legend for predictions
            cmap = {1: [*np.asarray([62, 33, 10])/255, 1.0],
                    2: [*np.asarray([129, 158, 131])/255, 1.0],
                    3: [1., 1., 1., 1.],
                    4: [*np.asarray([184, 126, 120]) / 255, 1.0]}
            labels = {1: 'TP', 2: 'FP', 3: 'TN', 4: 'FN'}
            pred_patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]

            # iterate through each slice
            for i, (slice_2d, gt_2d, tumor_2d) in enumerate(zip(orig_img_3d, ground_truth_3d, tumor_mask_3d)):
                # skip slices that contain no bladder
                if i in empty_mask_frames:
                    continue

                fig, axs = plt.subplots(int(np.ceil(size / 2)), 2, figsize=(10, 10))
                axs[0, 0].imshow(slice_2d, cmap=raw_img_cmap)
                axs[0, 0].set_title('Original')
                axs[0, 0].set_xlabel(x_label)
                axs[0, 0].set_ylabel(y_label)
                # FIXME when gt bladder and tumor overlap
                axs[0, 1].imshow(centre_crop(gt_2d + tumor_2d/2, crop_size), cmap=gt_cmap, interpolation='none', alpha=1.0)
                axs[0, 1].set_title('Ground Truth Mask\nGTSum: %.0f' % gt_2d.sum())
                axs[0, 1].legend(handles=gt_patches, loc=4)

                for idx, alg_name in enumerate(show_mask_algos):
                    idx += 2
                    # convert to 2d indices
                    a = int(idx / 2)
                    b = int(idx % 2)

                    slice_mask = patient_dict[alg_name][i]
                    slice_dice = 0. if slice_mask.sum() == 0 else dice_score(gt_2d, slice_mask)
                    title = "%s\nDice: %.04f, PredSum: %.0f" % (alg_name, slice_dice, slice_mask.sum())
                    tumor_pred_overlap = slice_mask[tumor_2d == 1].sum()
                    if tumor_pred_overlap != 0:
                        title += "\nTumor Pred Overlap = %.0f" % tumor_pred_overlap

                    axs[a, b].imshow(centre_crop(gt_2d, crop_size), cmap=gt_cmap, interpolation='none', alpha=1.0)
                    axs[a, b].imshow(centre_crop(slice_mask, crop_size), cmap=pred_cmap, interpolation='none', alpha=0.5)
                    axs[a, b].set_title(title)
                    axs[a, b].set_xticks([])
                    axs[a, b].set_yticks([])

                fig.suptitle("Patient ID: %s\nOrder: %.0f, Depth axis: %s" % (patient, i, "z"))
                # add legend for the prediction masks
                plt.legend(handles=pred_patches, loc=4, borderaxespad=0.)
                # adjust spacing
                plt.tight_layout()
                plt.subplots_adjust(top=0.90)

                if save_figs:
                    # save fig to disk
                    fig.savefig(os.path.join(mask_dir, "z", patient + "-" + str(fig_num) + ".png"), format="png")
                    plt.close(fig)
                    fig_num += 1

                else:
                    plt.show()
                    plt.close('all')

    print("\nDone iterating through patients...\n")

    # aggregate dice scores for all edge detection algorithms
    algos_dict = {}
    for p in results_dict.keys():
        for a in results_dict[p]["algos"]:
            if a not in algos_dict.keys():
                algos_dict[a] = {"dice": [results_dict[p]["algos"][a]["dice"]],
                                 "tumor_pred_overlap": [results_dict[p]["algos"][a]["tumor_pred_overlap"]]}
            else:
                algos_dict[a]["dice"].append(results_dict[p]["algos"][a]["dice"])
                algos_dict[a]["tumor_pred_overlap"].append(results_dict[p]["algos"][a]["tumor_pred_overlap"])

    # compute average scores across patients
    for alg_name in algos_dict.keys():
        print("{}\n\tmean dice score: {}\n\tmean tumor pred overlap: {}".format(alg_name,
                                                                                np.mean(algos_dict[alg_name]["dice"]),
                                                                                np.mean(algos_dict[alg_name]["tumor_pred_overlap"])))

    # save results to disk
    with open(os.path.join(DATA_DIR, 'mask_prediction_results_dict_' + run_name + "_" + bladder_frame_mode + '.pk'), 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plots histograms of the dice scores for each algorithm
    if show_hist:
        # set threshold below which lower values will be part of the same bin
        min_threshold = 0.7
        # TODO: add text which specifies number of scores below min_threshold
        # plot hist of dice scores averaged across scans
        figsize = (6, round(len(algos_dict)/5 * 9))
        fig1, axs1 = plt.subplots(len(algos_dict.keys()), figsize=figsize, sharex=True)
        for idx, alg_name in enumerate(algos_dict.keys()):
            # put everything below min threshold in the same bin
            dist = np.asarray(algos_dict[alg_name]["dice"])
            dist_mean = np.mean(dist)
            dist[dist < min_threshold] = min_threshold

            axs1[idx].hist(dist)
            axs1[idx].set_title("%s\nMean Dice Score = %.04f, Mean Tumor Pred Overlap" % (alg_name, float(dist_mean)))

            if idx == len(algos_dict.keys()) - 1:
                # generate x tick labels
                labels = list((np.asarray(axs1[idx].get_xticks()) * 100).astype(int) / 100)
                labels = ["< " + str(min_threshold) if label == min_threshold else label for label in labels]
                axs1[idx].set_xticklabels(labels)

        plt.suptitle("Histogram of Dice Scores Across Scans\nN = %.0f" % len(dist))
        plt.xlabel("Dice score")
        # adjust spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        if save_figs:
            fig1.savefig(os.path.join(mask_dir, "dice_scan_hist_" + run_name + "_" + bladder_frame_mode + ".png"), format="png")
        plt.show()
        plt.close('all')


if __name__ == '__main__':
    # get predictions on validation set
    classifier_results = pickle.load(open("results_dict_pred_frame_finder_50x50.pk", 'rb'), encoding='bytes')
    # specify classifier
    clf = 'svc-rbf'
    # get model predictions
    model_dict = classifier_results['clf'][clf]

    # iterate through each fold and get the corresponding patients keys and predictions
    preds = {}
    for fold in model_dict.keys():
        for patient in model_dict[fold]['val']:
            arr = model_dict[fold]['val'][patient]['pred']
            arr = np.where(arr != 0)[0]
            preds[patient] = arr

    # add arbitrary number of edge detection algorithms in the args
    generate_mask_predictions(CannyMask,
                              SobelMask,
                              MarchSquaresMask,
                              EnsembleMeanMask,
                              dataset="image_dataset", slice_axes=['x', 'y', 'z'], run_name="",
                              show_mask=False, show_hist=True, save_figs=True, bladder_frame_mode="pred", pred_dict=preds)
