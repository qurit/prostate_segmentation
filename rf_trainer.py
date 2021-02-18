import os
import pickle
import tqdm
import json
import numpy as np
from itertools import groupby, count
import torch
from utils import centre_crop
from scipy import stats

from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

DATA_DIR = "data"
SEED = 69
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


def main():
    # load up features and labels
    print("Loading features and labels into memory...")
    X = pickle.load(open(os.path.join(DATA_DIR, "train_images_192.pk"), 'rb'), encoding='bytes')
    y = pickle.load(open(os.path.join(DATA_DIR, "train_labels_192.pk"), 'rb'), encoding='bytes')
    print("Raw\nX size: {}\ny size: {}".format(X.shape, y.shape))

    # get test data
    X_test = pickle.load(open(os.path.join(DATA_DIR, "test_images_192.pk"), 'rb'), encoding='bytes')
    y_test = pickle.load(open(os.path.join(DATA_DIR, "test_labels_192.pk"), 'rb'), encoding='bytes')
    print("Raw\nX_test size: {}\ny_test size: {}".format(X_test.shape, y_test.shape))
    print("Done\n")

    # take out 4 random patients from train set and move it into the test set
    rand_train_idx = np.round(np.random.random(4) * 59).astype(int)  # should be [17 48 21 47]
    print("Removing training examples:", rand_train_idx)
    with open(os.path.join(DATA_DIR, 'image_dataset/global_dict.json')) as f:
        data_dict = json.load(f)
    move_patient_ids = [list(data_dict.keys())[i] for i in rand_train_idx]
    print("Corresponding patient ids:", move_patient_ids)  # should be ['PSMA-01-664', 'PSMA-01-669', 'PSMA-01-211', 'PSMA-01-732']

    # update train set
    move_X = X[rand_train_idx]
    move_y = y[rand_train_idx]
    X = np.delete(X, rand_train_idx, axis=0)
    y = np.delete(y, rand_train_idx, axis=0)
    # update test set
    # X_test = torch.cat((X_test, move_X))  # these are different dimensions so just ignore for now
    # y_test = y_test.append(move_y)

    # take a center crop
    crop_size = (50, 50)  # TODO: try 50x50, 100x100 and 25x25
    # pass in a copied image object, otherwise orig_img gets modified
    X = centre_crop(X, (*np.shape(X)[0:2], *crop_size))
    print("\nNew X size", X.shape)
    # X_test = centre_crop(X_test, (*np.shape(X_test)[0:2], *crop_size)) FIXME: ignoring test data for now

    # flatten X such that shape is (samples, features) and y such that shape is (samples)
    X = torch.flatten(X, start_dim=0, end_dim=1)
    X = torch.flatten(X, start_dim=1).numpy()
    y = torch.flatten(y, start_dim=0).numpy()

    print("\nDone preprocessing data!\n----------------------")

    print("X shape", np.shape(X))
    print("y shape", np.shape(y))
    print("Baseline accuracy", compute_baseline_score(y), "\n")
    print("X_test shape", np.shape(X_test))
    print("X_test_2 shape", np.shape(move_X))
    print("y_test shape", np.shape(y_test))
    print("y_test_2 shape", np.shape(move_y))
    print("Baseline accuracy", compute_baseline_score(y_test), "\n")

    # remove mean and scale to unit variance
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)  # TODO: make sure test features are also transformed

    # generate groups to split train and val sets across patients
    groups = np.tile(np.arange(0, 55), (100, 1)).flatten("F")
    # expect train:val to be 44:11 across 5 different folds
    group_kfold = GroupKFold(n_splits=5)
    print("Start training...\n")

    # define list of classifiers
    clfs = {"log-reg-l2": LogisticRegression(random_state=SEED, penalty="l2", class_weight='balanced', solver="sag", n_jobs=-1, max_iter=400),  # TODO: see if this helps
            "log-reg-l1": LogisticRegression(random_state=SEED, penalty="l1", class_weight='balanced', solver="saga", n_jobs=-1, max_iter=400),
            "svc-linear": SVC(kernel='linear', class_weight='balanced'),
            "svc-poly": SVC(kernel='poly', class_weight='balanced'),
            "svc-rbf": SVC(kernel='rbf', class_weight='balanced')}
            # FIXME "svr-linear": SVR(kernel='rbf')}  # no class_weight='balanced' for SVR...

    # "rf": RandomForestClassifier(max_samples=1, max_depth=400, random_state=0, n_jobs=-1, max_features="log2",
    #                                      min_samples_leaf=50, oob_score=True, class_weight="balanced")

    results_dict = {"baseline": [], "val_size": [], "num_pred_all_zeros": 0}
    split_num = 0
    all_val_idx = []
    clfs_keys = list(clfs.keys())
    clfs_keys.append("ensemble")

    for train_idx, val_idx in tqdm.tqdm(group_kfold.split(X, y, groups), total=group_kfold.get_n_splits(), desc="k-fold"):
        # make sure val_idx have never been seen
        assert (len(np.intersect1d(all_val_idx, val_idx)) == 0)
        print("\n#########################\n", "Split #", split_num, "\n")
        split_num += 1
        print("train_idx:", train_idx)
        print("val_idx:", val_idx)
        print("train size:", np.shape(train_idx)[0])
        print("val size:", np.shape(val_idx)[0])
        baseline = compute_baseline_score(y[val_idx])
        print("val baseline", baseline)
        results_dict["baseline"].append(compute_baseline_score(y[val_idx]))
        results_dict["val_size"].append(np.shape(val_idx)[0])
        all_val_idx.extend(val_idx)

        # array to store preds from each clf for ensemble pred
        ensemble = []

        for clf_name in tqdm.tqdm(clfs_keys, desc="clf"):
            print("\n\nClassifier:", clf_name)
            if clf_name == "ensemble":
                y_pred = np.round(np.mean(ensemble, axis=0)).astype(int)

            else:
                clf = clfs[clf_name].fit(X[train_idx], y[train_idx])
                # show train acc
                print("\nTraining acc:", np.sum(y[train_idx] == clf.predict(X[train_idx])) / len(train_idx))

                # compute and save validation results
                y_pred = clf.predict(X[val_idx])
                ensemble.append(y_pred)

            # pruning step
            y_pred_reshaped = y_pred.reshape(int(len(val_idx)/100), 100)
            new_preds = np.zeros_like(y_pred_reshaped)
            # get the longest contiguous prediction of ones in y_pred, thank you https://stackoverflow.com/a/44392621
            c = count()
            for pat, y_pred_pat in enumerate(y_pred_reshaped):
                try:
                    longest_stretch_of_1s = max((list(g) for _, g in groupby(np.argwhere(y_pred_pat == 1).flatten(), lambda x: x-next(c))), key=len)
                    new_preds[pat][longest_stretch_of_1s] = 1
                except:
                    print("\033[91m" + "WARNING: FOUND A PATIENT WHICH WAS PREDICTED ALL 0s")
                    print("y_pred", y_pred_pat)
                    print("y_gt", y[val_idx][pat*100:(pat+1)*100], "\n\033[0m")
                    results_dict["num_pred_all_zeros"] += 1
                    continue
            new_preds = new_preds.ravel()

            # show prediction for random patient
            rand_idx = int(np.random.random() * len(y_pred)/100)
            print("rand_idx", rand_idx)
            print("y_pred", y_pred[rand_idx*100:(rand_idx+1)*100])
            print("y_gt", y[val_idx][rand_idx*100:(rand_idx+1)*100])
            conf_matrix, acc = get_pred_results(y[val_idx], y_pred)

            print("\n\033[92m****Results with pruned predictions****")
            conf_matrix_p, acc_p = get_pred_results(y[val_idx], new_preds)
            print("\033[0m")

            if clf_name not in results_dict.keys():
                results_dict[clf_name] = {"acc": [acc],
                                          "tn": [conf_matrix[0][0]], "fn": [conf_matrix[1][0]],
                                          "tp": [conf_matrix[1][1]], "fp": [conf_matrix[0][1]],

                                          "acc_p": [acc_p],
                                          "tn_p": [conf_matrix_p[0][0]], "fn_p": [conf_matrix_p[1][0]],
                                          "tp_p": [conf_matrix_p[1][1]], "fp_p": [conf_matrix_p[0][1]]}
            else:
                results_dict[clf_name]["acc"].append(acc)
                results_dict[clf_name]["tn"].append(conf_matrix[0][0])
                results_dict[clf_name]["fn"].append(conf_matrix[1][0])
                results_dict[clf_name]["tp"].append(conf_matrix[1][1])
                results_dict[clf_name]["fp"].append(conf_matrix[0][1])

                results_dict[clf_name]["acc_p"].append(acc_p)
                results_dict[clf_name]["tn_p"].append(conf_matrix_p[0][0])
                results_dict[clf_name]["fn_p"].append(conf_matrix_p[1][0])
                results_dict[clf_name]["tp_p"].append(conf_matrix_p[1][1])
                results_dict[clf_name]["fp_p"].append(conf_matrix_p[0][1])

    print("Done training!", "\n________________________\n")
    print("results_dict", results_dict)

    # save to disk
    with open('results_dict_fixed_data_' + str(crop_size[0]) + "x" + str(crop_size[1]) + '.pk', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nComputing mean accuracy for each algorithm...")
    # print mean scores of all classifiers
    for clf_name in clfs_keys:
        print("     ", clf_name, np.mean(results_dict[clf_name]['acc']))

    print("\n****Results with pruned predictions****")
    for clf_name in clfs_keys:
        print("     ", clf_name, np.mean(results_dict[clf_name]['acc_p']))

    print("\033[91m\nNumber of patients which were predicted all 0s:", results_dict["num_pred_all_zeros"], "\n\033[0m")

    # evaluate on test set
    # fit
    # clf.fit(X, y)
    #
    # # predict
    # y_pred = clf.predict(X_test)
    #
    # # get mean accuracy
    # print("Mean test accuracy", clf.score(X_test, y_test))

    # priority list
    #  1. better way to visually compare y_pred with y_gt, print bladder frame range for each
    #  3. ensemble, only at test time
    #  4. probabilities of predictions

    #  4. after each fit on the train set, run edge detection on the val set and then save out dice score

    # TODO: could try PCA then knn


if __name__ == '__main__':
    main()