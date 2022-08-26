import os
import random
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Sampler

from scipy.ndimage import distance_transform_edt as eucl_distance


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # set to True if inputs are the same size during training
    torch.backends.cudnn.deterministic = True  # may result in a slowdown if set to True


def plot_loss(path, storage):
    plt.figure()
    train_loss = np.asarray(storage.history("training_loss").values())
    val_loss = np.asarray(storage.history("val_loss").values())
    plt.plot(train_loss[:, 1], train_loss[:, 0], label="Training")
    plt.plot(val_loss[:, 1], val_loss[:, 0], label="Validation")
    plt.xlabel("Iteration #")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Model loss")
    plt.savefig(path)


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


def expand_as_one_hot(input, C, ignore_index=None):
    """
    original code from
    https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/utils.py

    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None, dtype=None) -> np.ndarray:
    """
    original code from
    https://github.com/LIVIAETS/boundary-loss/blob/8f4457416a583e33cae71443779591173e27ec62/utils.py#L260
    """
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) *\
                        negmask - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


class TrainingSampler(Sampler):
    """
    original code from
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/samplers/distributed_sampler.py#L15

    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from torch.arange(self._size).tolist()
