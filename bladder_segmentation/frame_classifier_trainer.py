import os
import pickle
import tqdm
import json
import numpy as np
from scipy import stats
import pydicom as dicom
from itertools import groupby, count

from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

from utils import centre_crop
from dicom_code.contour_utils import parse_dicom_image

DATA_DIR = "data"
FRAMES_PER_PATIENT = 100  # TODO: make adaptive
SEED = 69
CROP_SIZE = 50
K_FOLD_N_SPLITS = 5
np.random.seed(SEED)


def compute_baseline_score(y):
    mode = stats.mode(y)[0][0]
    pred = np.zeros_like(y) + mode

    return np.sum(y == pred)/len(y)


def get_pred_results(y_gt, y_pred):
    conf_matrix = confusion_matrix(y_gt, y_pred)
    print("Confusion Matrix\n", conf_matrix)
    print(classification_report(y_gt, y_pred))
    acc = np.sum(y_gt == y_pred) / len(y_pred)
    print("Validation acc:", acc)

    return conf_matrix, acc


def frame_classifier_trainer(dataset="image_dataset", save_results=True, run_name=""):
    """
    Trains a bladder frame classifier for the bladder segmentation pipeline. Does 5-Fold cross validation and saves the
    predictions on the validation set for each fold. The train-val set split is 44:11

    Args:
        dataset: Name of the dataset inside DATA_DIR, "image_dataset" or "test_dataset"
        save_results: Option to save the results to disk
        run_name: Option to give a name to the training run which will add a suffix to the saved results dict

    Returns:

    """
    # load up features and labels
    print("Loading features and labels into memory...")
    with open(os.path.join(DATA_DIR, dataset, 'global_dict.json')) as f:
        data_dict = json.load(f)
    all_patient_keys = list(data_dict.keys())

    # take out 4 random patients from train set
    rand_train_idx = np.round(np.random.random(4) * 59).astype(int)  # should be [17 48 21 47]
    print("Removing training examples:", rand_train_idx)
    to_remove_patient_keys = [all_patient_keys[i] for i in rand_train_idx]
    print("Corresponding patient ids:", to_remove_patient_keys)  # should be ['PSMA-01-664', 'PSMA-01-669', 'PSMA-01-211', 'PSMA-01-732']

    train_patient_keys = [patient_key for idx, patient_key in enumerate(all_patient_keys) if idx not in rand_train_idx]
    assert (len(np.intersect1d(train_patient_keys, to_remove_patient_keys)) == 0)
    # convert to dict
    train_patient_idx_to_key_map = {}
    for idx, patient_key in enumerate(train_patient_keys):
        train_patient_idx_to_key_map[idx] = patient_key

    X_train = []
    y_train = []

    # specify which frames to consider from scan
    frame_range = np.arange(0, FRAMES_PER_PATIENT)

    for patient in tqdm.tqdm(train_patient_keys, desc="Patient keys"):
        scan = data_dict[patient]['PT']
        bladder = scan['rois']['Bladder']
        bladder_frames = [frame for frame, contour in enumerate(bladder) if contour != []]

        # get all frame file paths in bladder range
        frame_fps = [os.path.join(DATA_DIR, scan['fp'], str(frame) + '.dcm') for frame in frame_range]

        # generate 3d image from entire bladder frame range
        img_3d = np.asarray([parse_dicom_image(dicom.dcmread(fp)) for fp in frame_fps])
        assert (len(np.shape(img_3d)) == 3)  # make sure image is 3d
        X_train.append(img_3d)

        # generate labels
        y_train.append([1 if frame_idx in bladder_frames else 0 for frame_idx in frame_range])

    # print info about dataset
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    print("Labels size: {}\nImages size: {}".format(y_train.shape, X_train.shape))
    print("\nDone loading data!\n----------------------")

    # take a center crop
    crop_size = (CROP_SIZE, CROP_SIZE)
    print("\nTaking centre crop of size: {}...".format(crop_size))
    # pass in a copied image object, otherwise orig_img gets modified
    X_train = centre_crop(X_train, (*np.shape(X_train)[0:2], *crop_size))
    print("New X size", X_train.shape)

    # flatten X such that shape is (patients, frames, features)
    print("\nFlattening features...")
    X_train = np.reshape(X_train, (*X_train.shape[0:2], X_train.shape[2]*X_train.shape[3]))

    # remove mean and scale to unit variance
    print("\nApplying standard scaling...\n")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)  # TODO: make sure test features are also transformed

    print("X shape", X_train.shape)
    print("y shape", y_train.shape)
    print("Baseline accuracy", compute_baseline_score(y_train), "\n")
    print("\nDone preprocessing data!\n----------------------")

    print("Start training...\n")
    # define list of classifiers
    clfs = {"log-reg-l2": LogisticRegression(random_state=SEED, penalty="l2", class_weight='balanced', solver="sag", n_jobs=-1, max_iter=400),
            "log-reg-l1": LogisticRegression(random_state=SEED, penalty="l1", class_weight='balanced', solver="saga", n_jobs=-1, max_iter=400),
            "svc-linear": SVC(kernel='linear', class_weight='balanced'),
            "svc-poly": SVC(kernel='poly', class_weight='balanced'),
            "svc-rbf": SVC(kernel='rbf', class_weight='balanced')}  # TODO: drop 2 of these
    clfs_keys = list(clfs.keys())
    clfs_keys.append("ensemble")

    # expect train:val to be 44:11 across 5 different folds
    kf = KFold(n_splits=K_FOLD_N_SPLITS, random_state=SEED, shuffle=True)

    results_dict = {"baseline": [], "val_size": [], "num_pred_all_zeros": 0, "clf": {}}
    kf_split_num = 0
    all_val_idx = []

    for train_idx, val_idx in tqdm.tqdm(kf.split(list(train_patient_idx_to_key_map.keys())), total=kf.get_n_splits(), desc="k-fold"):
        # make sure val_idx have never been seen
        assert (len(np.intersect1d(all_val_idx, val_idx)) == 0)
        print("\n#########################\n", "Split #", kf_split_num, "\n")
        kf_split_num += 1
        print("train_idx:", train_idx)
        print("val_idx:", val_idx)
        print("train size:", np.shape(train_idx)[0])
        print("val size:", np.shape(val_idx)[0])
        baseline = compute_baseline_score(y_train[val_idx])
        print("val baseline", baseline)
        results_dict["baseline"].append(compute_baseline_score(y_train[val_idx]))
        results_dict["val_size"].append(np.shape(val_idx)[0])
        all_val_idx.extend(val_idx)

        # find the patient keys corresponding to these indices
        curr_patient_keys_train = [train_patient_idx_to_key_map[k] for k in train_idx]
        curr_patient_keys_val = [train_patient_idx_to_key_map[k] for k in val_idx]
        assert (len(np.intersect1d(curr_patient_keys_train, curr_patient_keys_val)) == 0)

        # get X and y for this fold and reshape train data to (samples, features) and (samples) respectively
        curr_X_train = X_train[train_idx]
        curr_X_train = np.reshape(curr_X_train, (curr_X_train.shape[0]*curr_X_train.shape[1], curr_X_train.shape[2]))
        curr_y_train = y_train[train_idx].flatten()

        curr_X_val = X_train[val_idx]
        curr_y_val = y_train[val_idx]

        # array to store preds from each clf for ensemble pred_dict
        ensemble = []

        for clf_name in tqdm.tqdm(clfs_keys, desc="clf"):
            if clf_name not in results_dict["clf"].keys():
                results_dict["clf"][clf_name] = {kf_split_num: {}}
            else:
                results_dict["clf"][clf_name][kf_split_num] = {}
            curr_clf_dict = {"train": curr_patient_keys_train, "val": {}}

            print("\n\nClassifier:", clf_name)
            if clf_name == "ensemble":
                all_y_preds = np.round(np.mean(ensemble, axis=0)).astype(int)
                for idx, (y_pred, y_val, val_patient_key) in \
                        enumerate(zip(all_y_preds.reshape(len(val_idx), curr_y_val.shape[1]), curr_y_val, curr_patient_keys_val)):
                    pruned_pred = np.zeros_like(y_pred)
                    # prune y_pred, get the longest contiguous prediction of ones in y_pred, thank you https://stackoverflow.com/a/44392621
                    c = count()
                    longest_stretch_of_1s = max((list(g) for _, g in groupby(np.argwhere(y_pred == 1).flatten(), lambda x: x - next(c))), key=len)
                    pruned_pred[longest_stretch_of_1s] = 1
                    # store in dict
                    curr_clf_dict["val"][val_patient_key] = {"pred_dict": pruned_pred, "gt": y_val, "raw_preds": y_pred}

            else:
                clf = clfs[clf_name].fit(curr_X_train, curr_y_train)
                # show train acc
                print("\nTraining acc:", np.sum(curr_y_train == clf.predict(curr_X_train)) / len(train_idx))

                # get predictions for each patient in the val set
                all_y_preds = []
                for idx, (X_val, y_val, val_patient_key) in enumerate(zip(curr_X_val, curr_y_val, curr_patient_keys_val)):
                    y_pred = clf.predict(X_val)
                    pruned_pred = np.zeros_like(y_pred)
                    # prune y_pred, get the longest contiguous prediction of ones in y_pred, thank you https://stackoverflow.com/a/44392621
                    c = count()
                    try:
                        longest_stretch_of_1s = max((list(g) for _, g in groupby(np.argwhere(y_pred == 1).flatten(),
                                                                                 lambda x: x - next(c))), key=len)
                        pruned_pred[longest_stretch_of_1s] = 1
                    except ValueError:
                        print("\033[91m" + "WARNING: FOUND A PATIENT WHICH WAS PREDICTED ALL 0s" + "\n\033[0m")
                        results_dict["num_pred_all_zeros"] += 1

                    all_y_preds.extend(pruned_pred)
                    # store in dict
                    curr_clf_dict["val"][val_patient_key] = {"pred_dict": pruned_pred, "gt": y_val, "raw_preds": y_pred}

                ensemble.append(all_y_preds)

            print("\n\033[92m****Results with pruned predictions****")
            conf_matrix, acc = get_pred_results(curr_y_val.flatten(), all_y_preds)
            print("\033[0m")

            # add val results to dict
            aggregated_val_results = {"acc": acc,
                                      "tn": conf_matrix[0][0], "fn": conf_matrix[1][0],
                                      "tp": conf_matrix[1][1], "fp": conf_matrix[0][1]}
            results_dict["clf"][clf_name][kf_split_num] = {**curr_clf_dict, **aggregated_val_results}
            # TODO: logic to save out model

    print("Done training!", "\n________________________\n")

    if save_results:
        # save to disk
        with open('results_dict_' + run_name + '_' + str(crop_size[0]) + "x" + str(crop_size[1]) + '.pk', 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nComputing mean accuracy for each classifier across KFolds...")
    # print mean scores of all classifiers
    for clf_name in clfs_keys:
        acc_arr = [results_dict["clf"][clf_name][fold]["acc"] for fold in results_dict["clf"][clf_name]]
        print("     ", clf_name, np.mean(acc_arr))

    print("\033[91m\nNumber of patients which were predicted all 0s:", results_dict["num_pred_all_zeros"], "\n\033[0m")

    # TODO: add inference on test data


if __name__ == '__main__':
    frame_classifier_trainer(dataset="image_dataset", save_results=False, run_name="")
