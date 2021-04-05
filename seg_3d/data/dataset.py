# original code from https://github.com/cosmic-cortex/pytorch-UNet/blob/master/unet/dataset.py
import os
import json
import glob
import numpy as np
import torch
import pydicom as dicom
from collections import UserList
from typing import List, Tuple

from skimage import io

from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Callable

from dicom_code.contour_utils import parse_dicom_image
from utils import contour2mask, centre_crop


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, crop=(256, 256), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0, long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        # transforming to tensor
        image = F.to_tensor(image)
        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask


class ImageToImage3D(Dataset):
    """
    Reads the dataset dict and applies the augmentations on images and masks.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- global_dict.json
              |-- patient001
                  |-- CT
                      |-- 1.dcm
                      |-- 2.dcm
                      |-- ...
                  |-- PT
                      |-- 1.dcm
                      |-- 2.dcm
                      |-- ...
              |-- ...
        modality: specifies modality of scan
        rois: list of region of interests
        patient_keys: optional arg to specify patients, if None then use all patients from dataset
        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, modality: str, rois: List[str], num_slices: int, crop_size: Tuple[int],
                 patient_keys: List[str] = None, joint_transform: Callable = None, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.modality = modality
        self.patient_keys = patient_keys
        self.rois = rois
        self.num_slices = num_slices
        self.crop_size = crop_size
        self.one_hot_mask = one_hot_mask

        with open(os.path.join(dataset_path, "global_dict.json")) as file_obj:
            self.dataset_dict = json.load(file_obj)

        # if no patients specified then select all from dataset
        if patient_keys is None:
            self.patient_keys = self.dataset_dict.keys()

        # self.all_frame_fps = {patient: glob.glob(patient_data[self.modality]['fp']+'/*.dcm') for patient,
        #                                                                      patient_data in self.dataset_dict.items()}
        self.all_frame_fps = {patient: glob.glob("data/" + self.dataset_dict[patient][self.modality]['fp'] + "/*.dcm")
                              for patient in self.patient_keys}

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            # to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (torch.from_numpy(x), torch.from_numpy(y))

    def __len__(self):
        return len(self.patient_keys)

    def __getitem__(self, idx):
        patient = list(self.patient_keys)[idx]
        frame_fps = sorted(self.all_frame_fps[patient], key=lambda x: int(os.path.basename(x).split('.')[0]))

        # read image
        image = np.asarray([parse_dicom_image(dicom.dcmread(fp)) for fp in frame_fps])

        # read mask image
        mask = self.get_mask(patient, image)

        image = centre_crop(image, (image.shape[0], *self.crop_size))[:self.num_slices]
        mask = centre_crop(mask, (mask.shape[0], *self.crop_size))[:self.num_slices]

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        mask = mask.unsqueeze(0)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, *mask.shape[1:])).scatter_(0, mask.long(), 1)

        return {
            "image": image.unsqueeze(0).float(),
            "gt_mask": mask.float(),
            "patient": patient
        }

    def get_mask(self, patient, image):
        # FIXME: assumes roi is size 1
        img_size = np.shape(image)
        roi = self.dataset_dict[patient][self.modality]['rois'][self.rois[0]]
        return np.asarray(
            [
                contour2mask(roi[frame], img_size[1:3])
                for frame in range(len(self.all_frame_fps[patient]))
            ]
        )


class Image2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them. As opposed to ImageToImage2D, this
    reads a single image and requires a simple augmentation transform.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as a prediction
           dataset.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        transform: augmentation transform. If bool(joint_transform) evaluates to False,
            torchvision.transforms.ToTensor will be used.
    """

    def __init__(self, dataset_path: str, transform: Callable = None):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.images_list = os.listdir(self.input_path)

        if transform:
            self.transform = transform
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image = io.imread(os.path.join(self.input_path, image_filename))

        # correct dimensions if needed
        image = correct_dims(image)

        image = self.transform(image)

        return image, image_filename
