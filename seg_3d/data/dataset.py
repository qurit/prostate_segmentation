# original code from https://github.com/cosmic-cortex/pytorch-UNet/blob/master/unet/dataset.py
import glob
import json
import logging
import os
from typing import Callable
from typing import List, Tuple

import elasticdeform
import numpy as np
import pydicom as dicom
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from dicom_code.contour_utils import parse_dicom_image
from utils import contour2mask, centre_crop


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
    """

    def __init__(self, test=False, crop=(100, 100), p_flip=0.5, deform_sigma=None, div_by_max=True):
        self.crop = crop
        self.p_flip = p_flip
        self.div_by_max = div_by_max
        self.test = test

        if deform_sigma:
            self.deform = lambda x, y: \
                elasticdeform.deform_random_grid([x, *y], sigma=deform_sigma,
                                                 order=[3, *[0] * len(y)])  # order must be 0 for mask arrays
        else:
            self.deform = lambda x, y: [x, *y]

    def __call__(self, image, masks):

        # divide by scan max
        if self.div_by_max:
            image = image / np.max(image)

        sample_data = self.deform(image, masks)
        image, masks = sample_data[0], sample_data[1:]

        # fix channel for background
        bg = np.ones_like(image)
        for m in masks[1:]:
            bg[bg == m] = 0
        masks[0] = bg

        # transforming to tensor
        image = torch.Tensor(image)
        mask = torch.Tensor(np.stack(masks, axis=0).astype(int))

        # random crop
        if self.crop:
            if not self.test:
                i, j, h, w = T.RandomCrop.get_params(image, self.crop)
                image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
            else:
                image, mask = T.CenterCrop(self.crop[0])(image), T.CenterCrop(self.crop[0])(mask)

        if self.p_flip and np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

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
    """

    def __init__(self, dataset_path: str, modality: str, rois: List[str], num_slices: int, crop_size: Tuple[int],
                 joint_transform: Callable, patient_keys: List[str] = None, num_patients: int = None) -> None:
        self.dataset_path = dataset_path
        self.modality = modality
        self.patient_keys = patient_keys
        self.rois = rois
        self.num_slices = num_slices
        self.crop_size = crop_size
        self.num_patients = num_patients  # used for train-val-test split
        self.logger = logging.getLogger(__name__)

        with open(os.path.join(dataset_path, "global_dict.json")) as file_obj:
            self.dataset_dict = json.load(file_obj)

        # if no patients specified then select all from dataset
        if patient_keys is None:
            self.patient_keys = list(self.dataset_dict.keys())

        # sample select patients if num_patients specified
        if num_patients is not None:
            selected_patients = np.random.choice(sorted(self.patient_keys), size=self.num_patients, replace=False)
            # keep track of excluded patients from selected
            self.excluded_patients = list(set(self.patient_keys) - set(selected_patients))
            self.patient_keys = selected_patients

        all_frame_fps = {patient: glob.glob('data/' + self.dataset_dict[patient][self.modality]['fp'] + "/*.dcm")
                         for patient in self.patient_keys}
        # sort the frame_fps based on number in the .dcm file name
        self.all_frame_fps = {patient: sorted(all_frame_fps[patient],
                                              key=lambda x: int(os.path.basename(x).split('.')[0]))[:self.num_slices]
                              for patient in self.patient_keys}

        self.joint_transform = joint_transform

    def __len__(self) -> int:
        return len(self.patient_keys)

    def __getitem__(self, idx) -> dict:
        patient = list(self.patient_keys)[idx]
        frame_fps = self.all_frame_fps[patient]

        # read image
        image = np.asarray([parse_dicom_image(dicom.dcmread(fp)) for fp in frame_fps]).astype(np.float32)
        # keep a copy of the unmodified image
        orig_image = np.copy(image)

        # read mask image
        masks_array = self.get_mask(patient, image)
        image = centre_crop(image, (self.num_slices, *self.crop_size))
        masks = [centre_crop(mask, (self.num_slices, *self.crop_size)) for mask in masks_array]

        # clip values if modality is CT, no preprocessing of values necessary for PET
        if self.modality == "CT":
            image[image > 150] = 150
            image[image < -150] = -150

        if self.joint_transform:
            image, mask = self.joint_transform(image, masks)

        return {
            "orig_image": orig_image,
            "image": image.unsqueeze(0).float(),
            "gt_mask": mask.float(),
            "patient": patient
        }

    def get_mask(self, patient, image) -> list:
        slice_size = np.shape(image)[1:3]

        # get specified roi data from dataset
        patient_rois = self.dataset_dict[patient][self.modality]['rois'].keys()
        roi_data = {roi_name: self.dataset_dict[patient][self.modality]['rois'][roi_name]
                    for roi_name in self.rois if roi_name in patient_rois}

        # build mask object for each roi
        mask = {
            roi_name: np.asarray(
                [contour2mask(roi_data[roi_name][frame], slice_size)
                 for frame in range(self.num_slices)]
            ) for roi_name in roi_data
        }

        if "Tumor" in self.rois and "Bladder" in self.rois:
            # call helper function to process tumor and bladder masks
            mask = self.process_bladder_tumor_mask(mask)
        else:
            # add logic here for other combinations of rois
            pass

        # make a dummy mask for the background channel, will be fixed in the transform step
        bg = np.zeros_like(list(mask.values())[0])

        return [bg, *list(mask.values())]

    @staticmethod
    def process_bladder_tumor_mask(mask) -> dict:
        tumor_keys = [x for x in mask.keys() if "Tumor" in x]
        tumor_mask = np.zeros_like(mask[tumor_keys[0]])

        # merge Tumor rois into a single channel
        for tum in tumor_keys:
            tumor_mask += mask[tum]
            del mask[tum]

        tumor_mask[tumor_mask > 1] = 1
        mask["Bladder"][tumor_mask == 1] = 0  # ensure there is no overlap in gt bladder mask

        # return new mask object just with Bladder and Tumor roi
        return {
            "Bladder": mask["Bladder"],
            "Tumor": tumor_mask
        }
