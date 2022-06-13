# Original code from https://github.com/alejandrodebus/Pytorch-Utils
import numpy as np


def partitions(number, k):
    """
    Distribution of the folds
    Args:
        number: number of patients
        k: folds number
    """
    n_partitions = np.ones(k) * int(number / k)
    n_partitions[0:(number % k)] += 1
    return n_partitions


def get_indices(n_splits, subjects, frames=1):
    """
    Indices of the set test
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    """
    l = partitions(subjects, n_splits)
    fold_sizes = l * frames
    indices = np.arange(subjects * frames).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size
        current = stop
        yield indices[int(start):int(stop)]


def k_folds(n_splits, subjects, frames=1):
    """
    Generates folds for cross validation
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    """
    indices = np.arange(subjects * frames).astype(int)
    for test_idx in get_indices(n_splits, subjects, frames):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx
