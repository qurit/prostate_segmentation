# original code from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/utils.py
# Copyright (c) 2018 Adrian Wolny
import os
import random
from typing import Optional
from zipfile import ZipFile

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Sampler


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


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    # logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        # logger.info(f"Current patch size: {shape}")
        model(patch)


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


def expand_as_one_hot(input, C, ignore_index=None):
    """
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


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


class TrainingSampler(Sampler):
    """
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


def zip_files_in_dir(dir_name, zip_file_name, to_ignore=None):
    # original code from
    # https://thispointer.com/python-how-to-create-a-zip-archive-from-multiple-files-or-directory/
    if to_ignore is None:
        to_ignore = ["__pycache__", "output"]  # by default ignore these directories
    filter_fn = lambda name: all(
        item not in name for item in to_ignore
    )

    with ZipFile(zip_file_name, 'w') as zip_obj:
        # Iterate over all the files in directory
        for folder_name, subfolders, filenames in os.walk(dir_name):
            if filter_fn(folder_name):
                for filename in filenames:
                    if filter_fn(filename):
                        # create complete filepath of file in directory
                        filePath = os.path.join(folder_name, filename)
                        # Add file to zip
                        zip_obj.write(filePath, filePath)

