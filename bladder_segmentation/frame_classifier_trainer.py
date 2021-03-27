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

from im_utils import centre_crop
from dicom_code.contour_utils import parse_dicom_image
from bladder_segmentation.pred_frame_finder import BladderFrameFinder


DATA_DIR = "data"
SEED = 69
np.random.seed(SEED)


def compute_baseline_score(y):
    mode = stats.mode(y)[0][0]
    pred = np.zeros_like(y) + mode

    return np.sum(y == pred)/len(y)


def get_pred_results(y_gt, y_pred, phase="Validation"):
    conf_matrix = confusion_matrix(y_gt, y_pred)
    print("Confusion Matrix\n", conf_matrix)
    print(classification_report(y_gt, y_pred))
    acc = np.sum(y_gt == y_pred) / len(y_pred)
    print(phase, "accuracy:", acc)

    return conf_matrix, acc


class BladderFrameClassifier:

    def __init__(self, frames_per_patient=100, crop_size=(50, 50), feature_means=None, feature_vars=None, **clfs: object):
        self.frames_per_patient = frames_per_patient
        self.crop_size = crop_size
        self.clfs = clfs
        self.feature_means = feature_means
        self.feature_vars = feature_vars

    def fit(self, run_name, dataset_name="image_dataset", k_fold_n_splits=5, ensemble_model=False):
        """
        Trains a bladder frame classifier for the bladder segmentation pipeline. Does KFold cross validation and saves the
        predictions on the validation set for each fold.

        Args:
            run_name: Name of the training run and directory which will store training outputs
            dataset_name: Name of the dataset inside DATA_DIR, "image_dataset" or "test_dataset"
            k_fold_n_splits: Number of splits for KFold cross validation. Default is 5-Fold so train-val set split is 44:11
            ensemble_model: Option to do ensembling by averaging the predictions across the classifiers

        Returns:
            model_dir: directory path where trained model and results dict are stored

        """
        # dir to save trained models and results
        model_dir = os.path.join("bladder_segmentation", "experiments", "models", run_name)
        os.makedirs(model_dir, exist_ok=True)

        # load up features and labels
        print("Loading features and labels into memory...")
        with open(os.path.join(DATA_DIR, dataset_name, 'global_dict.json')) as f:
            data_dict = json.load(f)
        all_patient_keys = list(data_dict.keys())

        # take out 4 random patients from train set
        rand_train_idx = np.round(np.random.random(4) * len(all_patient_keys)).astype(int)  # should be [17 48 21 47]
        print("Removing training examples:", rand_train_idx)
        removed_from_train_test_patient_keys = [all_patient_keys[i] for i in rand_train_idx]
        print("Corresponding patient ids:", removed_from_train_test_patient_keys)  # should be ['PSMA-01-664', 'PSMA-01-669', 'PSMA-01-211', 'PSMA-01-732']

        train_patient_keys = [patient_key for idx, patient_key in enumerate(all_patient_keys) if idx not in rand_train_idx]
        assert (len(np.intersect1d(train_patient_keys, removed_from_train_test_patient_keys)) == 0)
        # make a patient index to key map
        train_patient_idx_to_key_map = {}
        for idx, patient_key in enumerate(train_patient_keys):
            train_patient_idx_to_key_map[idx] = patient_key

        X_train, y_train = self.load_and_preprocess_data(data_dict, train_patient_keys)

        # remove mean and scale to unit variance
        print("\nApplying standard scaling...\n")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

        print("X shape", X_train.shape)
        print("y shape", y_train.shape)
        print("\nBaseline accuracy", compute_baseline_score(y_train))
        print("\nDone preprocessing data!\n----------------------")

        print("Start training...\n")
        # initialize bladder frame object
        bladder_frame_finder = BladderFrameFinder()
    
        clfs_keys = list(self.clfs.keys())
        if ensemble_model:
            clfs_keys.append("ensemble")  # NOTE: could instead use sklearn.ensemble.StackingClassifier

        # expect train:val to be 44:11 across 5 different folds
        kf = KFold(n_splits=k_fold_n_splits, random_state=SEED, shuffle=True)

        results_dict = {"feature_means": scaler.mean_, "feature_vars": scaler.var_, "baseline": [], "val_size": [], "clf": {}}
        self.feature_means = scaler.mean_
        self.feature_vars = scaler.var_

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
            print("val baseline:", baseline)
            results_dict["baseline"].append(baseline)
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

            # array to store preds from each clf for ensemble pred
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
                    all_y_preds_pruned = []
                    for y_pred, y_val, val_patient_key in zip(all_y_preds.reshape(len(val_idx), curr_y_val.shape[1]), curr_y_val, curr_patient_keys_val):
                        pred_frame, _ = bladder_frame_finder.find_bladder_frame(DATA_DIR, data_dict[val_patient_key]['PT'])
                        pruned_pred = self.prune_pred(y_pred, pred_frame)

                        all_y_preds_pruned.extend(pruned_pred)
                        # store in dict
                        curr_clf_dict["val"][val_patient_key] = {"pred": pruned_pred, "gt": y_val, "raw_preds": y_pred}
                    all_y_preds = all_y_preds_pruned

                else:
                    clf = self.clfs[clf_name].fit(curr_X_train, curr_y_train)
                    # show train acc
                    print("\nTraining acc:", np.sum(curr_y_train == clf.predict(curr_X_train)) / len(train_idx))

                    # get predictions for each patient in the val set
                    all_y_preds = []
                    for X_val, y_val, val_patient_key in zip(curr_X_val, curr_y_val, curr_patient_keys_val):
                        y_pred = clf.predict(X_val)
                        pred_frame, _ = bladder_frame_finder.find_bladder_frame(DATA_DIR, data_dict[val_patient_key]['PT'])
                        pruned_pred = self.prune_pred(y_pred, pred_frame)

                        all_y_preds.extend(pruned_pred)
                        # store in dict
                        curr_clf_dict["val"][val_patient_key] = {"pred": pruned_pred, "gt": y_val, "raw_preds": y_pred}

                    ensemble.append(all_y_preds)

                print("\n\033[92m****Results with pruned predictions****")
                conf_matrix, acc = get_pred_results(curr_y_val.flatten(), all_y_preds)
                print("\033[0m")

                # add val results to dict
                aggregated_val_results = {"acc": acc,
                                          "tn": conf_matrix[0][0], "fn": conf_matrix[1][0],
                                          "tp": conf_matrix[1][1], "fp": conf_matrix[0][1]}
                results_dict["clf"][clf_name][kf_split_num] = {**curr_clf_dict, **aggregated_val_results}

                # save to disk trained model after each kfold
                with open(os.path.join(model_dir, clf_name + str(kf_split_num) + '.pk'), 'wb') as handle:
                    pickle.dump(self.clfs[clf_name], handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Done training!", "\n________________________\n")

        # save to disk the results
        with open(os.path.join(model_dir, 'results_dict_' + str(self.crop_size[0]) + "x" + str(self.crop_size[1]) + '.pk'), 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save to disk all the trained models, currently cannot save out a single file for Ensemble
        # for clf_name in self.clfs:
        #     with open(os.path.join(model_dir, clf_name + '.pk'), 'wb') as handle:
        #         pickle.dump(self.clfs[clf_name], handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("\nComputing mean accuracy for each classifier across KFolds...")
        # print mean scores of all classifiers
        for clf_name in clfs_keys:
            acc_arr = [results_dict["clf"][clf_name][fold]["acc"] for fold in results_dict["clf"][clf_name]]
            print("     ", clf_name, np.mean(acc_arr))

        return model_dir

    def predict(self, results_dir, clf_name, dataset_name="test_dataset", patient_keys=None):
        # load models
        # clf = self.clfs[clf_name]

        # load up test data features and labels
        print("Loading features and labels into memory...")
        with open(os.path.join(DATA_DIR, dataset_name, 'global_dict.json')) as f:
            data_dict = json.load(f)

        if patient_keys is None:
            patient_keys = list(data_dict.keys())
        print("patient keys in test set:", patient_keys)

        X_test, y_test = self.load_and_preprocess_data(data_dict, patient_keys)

        # remove mean and scale to unit variance
        print("\nApplying standard scaling...\n")
        X_test = ((X_test.reshape(-1, X_test.shape[-1]) - self.feature_means) / self.feature_vars**0.5).reshape(X_test.shape)

        print("X shape", X_test.shape)
        print("y shape", X_test.shape)
        print("\nBaseline accuracy", compute_baseline_score(y_test))
        print("\nDone preprocessing data!\n----------------------")

        # run inference on each patient
        print("Run inference on test data...\n")

        # initialize bladder frame object
        bladder_frame_finder = BladderFrameFinder()

        # dict to store frame predictions for each patient
        preds_dict = {}

        # dict to store metric scores
        test_metrics = {"acc": [], "tn": [], "fn": [], "tp": [], "fp": []}

        for idx, patient in tqdm.tqdm(enumerate(patient_keys), total=len(patient_keys), desc="Patient keys"):
            print("\n######################\nPatient ID", patient)

            ensemble = []
            # get model from each fold
            for i in range(5):
                clf = self.clfs[clf_name + str(i + 1)]
                ensemble.append(clf.predict(X_test[idx]))
            print("\nensemble shape", np.shape(ensemble))

            y_pred = np.round(np.sum(ensemble, axis=0)).astype(int)
            y_pred[y_pred > 1] = 1

            pred_frame, _ = bladder_frame_finder.find_bladder_frame(DATA_DIR, data_dict[patient]['PT'])
            pruned_pred = self.prune_pred(y_pred, pred_frame)
            print("y_pred:        {}\n"
                  "y_pred_pruned: {}".format(y_pred, pruned_pred))

            preds_dict[patient] = pruned_pred

            print("\n\033[92m****Results with pruned predictions****")
            conf_matrix, acc = get_pred_results(y_test[idx].flatten(), pruned_pred, phase="Test")
            print("\033[0m")

            # add metric results to dict
            test_metrics["acc"].append(acc)
            test_metrics["tn"].append(conf_matrix[0][0])
            test_metrics["fn"].append(conf_matrix[1][0])
            test_metrics["tp"].append(conf_matrix[1][1])
            test_metrics["fp"].append(conf_matrix[0][1])

        print("Done running inference!", "\n________________________\n")

        # save to disk the results
        with open(os.path.join(results_dir, 'predictions_' + dataset_name + '.pk'), 'wb') as handle:
            pickle.dump(preds_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("\nComputing mean accuracy...")
        # print mean score
        print("     ", clf_name, np.mean(test_metrics["acc"]))

    def load_and_preprocess_data(self, data_dict, patient_keys):
        X = []
        y = []

        # specify which frames to consider from scan
        frame_range = np.arange(0, self.frames_per_patient)
        for patient in tqdm.tqdm(patient_keys, desc="Patient keys"):
            scan = data_dict[patient]['PT']
            bladder = scan['rois']['Bladder']
            bladder_frames = [frame for frame, contour in enumerate(bladder) if contour != []]

            # get all frame file paths in bladder range
            frame_fps = [os.path.join(DATA_DIR, scan['fp'], str(frame) + '.dcm') for frame in frame_range]

            # generate 3d image from entire bladder frame range
            img_3d = np.asarray([parse_dicom_image(dicom.dcmread(fp)) for fp in frame_fps])
            assert (len(np.shape(img_3d)) == 3)  # make sure image is 3d
            X.append(img_3d)

            # generate labels
            y.append([1 if frame_idx in bladder_frames else 0 for frame_idx in frame_range])

        # print info about dataset
        X = np.asarray(X)
        y = np.asarray(y)
        print("Labels size: {}\nImages size: {}".format(y.shape, X.shape))
        print("\nDone loading data!\n----------------------")
        # take a center crop
        print("\nTaking centre crop of size: {}...".format(self.crop_size))
        X = centre_crop(X, (*np.shape(X)[0:2], *self.crop_size))
        print("New X size", X.shape)
        # flatten X such that shape is (patients, frames, features)
        print("\nFlattening features...")
        X = np.reshape(X, (*X.shape[0:2], X.shape[2] * X.shape[3]))

        return X, y

    @staticmethod
    def prune_pred(y_pred, pred_frame):
        pruned_pred = np.zeros_like(y_pred)

        if y_pred.sum() == 0:
            print("\033[91m" + "WARNING: FOUND A PATIENT WHICH WAS PREDICTED ALL 0s" + "\n\033[0m")

        else:
            # prune y_pred, get the longest contiguous prediction of ones in y_pred, thank you https://stackoverflow.com/a/44392621
            c = count()
            longest_stretch_of_1s = [list(g) for _, g in
                                     groupby(np.argwhere(y_pred == 1).flatten(), lambda x: x - next(c))]

            # iterate through each of the candidate bladder frame ranges and find the one which contains pred_frame
            for lst in longest_stretch_of_1s:
                if pred_frame in lst:
                    # found a predicted range which contains pred_frame so update predictions
                    pruned_pred[lst] = 1
                    break

            # check if pruned_pred has been updated
            if pruned_pred.sum() == 0:
                print("\033[91m" + "WARNING: FOUND A PREDICTION WHICH DID NOT CONTAIN PRED_FRAME" + "\n\033[0m")
                pruned_pred[max(longest_stretch_of_1s, key=len)] = 1

        return pruned_pred


if __name__ == '__main__':
    # define list of classifiers
    # clfs = {"svc-rbf": SVC(kernel='rbf', class_weight='balanced')}

    # clfs = {"log-reg-l2": LogisticRegression(random_state=SEED, penalty="l2", class_weight='balanced', solver="sag", n_jobs=-1, max_iter=400),
    #         "log-reg-l1": LogisticRegression(random_state=SEED, penalty="l1", class_weight='balanced', solver="saga", n_jobs=-1, max_iter=400),
    #         "svc-linear": SVC(kernel='linear', class_weight='balanced'),
    #         "svc-poly": SVC(kernel='poly', class_weight='balanced')}

    # init classifier object
    # model = BladderFrameClassifier(frames_per_patient=100, crop_size=(50, 50), **clfs)

    # fit to train data
    run_name = "run-cross-val-test-1"
    # model_dir = model.fit(run_name=run_name, dataset_name="image_dataset", k_fold_n_splits=5)

    model_dir = os.path.join("bladder_segmentation/experiments/models", run_name)

    # get feature means and variances, TODO: use scikit-learn pipeline and add standard scalar
    classifier_results = pickle.load(open(os.path.join(model_dir, "results_dict_50x50.pk"), 'rb'), encoding='bytes')

    # specify model
    clf_name = 'svc-rbf'

    clfs = {}
    for j in range(5):
        # load classifier
        with open(os.path.join(model_dir, clf_name + str(j + 1) + '.pk'), "rb") as f:
            raw_clf = f.read()
        clfs[clf_name + str(j + 1)] = pickle.loads(raw_clf)

    # init classifier object
    model = BladderFrameClassifier(frames_per_patient=300, crop_size=(50, 50),
                                   feature_means=classifier_results["feature_means"],
                                   feature_vars=classifier_results["feature_vars"], **clfs)

    # run inference
    model.predict(model_dir, clf_name, dataset_name="test_dataset")

    model.predict(model_dir, clf_name, dataset_name="test_dataset",
                  patient_keys=['PSMA-01-664', 'PSMA-01-669', 'PSMA-01-211', 'PSMA-01-732'])
