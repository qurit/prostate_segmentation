import glob
import json
import logging
import os
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import elasticdeform
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from seg_3d.data.itk_image_resample import *
from seg_3d.utils.misc_utils import one_hot2dist
from seg_3d.utils.slice_builder import SliceBuilder
from utils import contour2mask, centre_crop


class JointTransform3D:
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

    def __init__(self, test: bool = False, crop: Tuple = None, p_flip: float = None, deform_sigma: float = None,
                 deform_points: Tuple or int = (3, 3, 3), div_by_max: bool = True, multi_scale: Tuple = None,
                 **kwargs):
        self.crop = crop
        self.p_flip = p_flip
        self.div_by_max = div_by_max
        self.test = test
        self.multi_scale = multi_scale  # min and max scaling factors

        if deform_sigma and not self.test:
            self.deform = lambda x, y: \
                elasticdeform.deform_random_grid([*x, *y], sigma=deform_sigma, points=deform_points,
                                                 order=[*[3] * len(x),
                                                        *[0] * len(y)])  # order must be 0 for mask arrays
        else:
            self.deform = lambda x, y: [*x, *y]

    def __call__(self, image: np.ndarray, masks: np.ndarray) -> Tuple[torch.tensor, torch.tensor]:

        # divide by scan max
        if self.div_by_max:
            image = image / np.max(image)

        # get number of channels in image
        img_channels = len(image)

        # apply elastic deformation if specified
        sample_data = self.deform(image, masks)
        image, masks = sample_data[0:img_channels], sample_data[img_channels:]

        bg = np.ones_like(masks[0])
        for m in masks:
            bg[bg == m] = 0
        masks = [bg, *masks]
        
        # transforming to tensor
        image = torch.Tensor(np.array(image))
        mask = torch.Tensor(np.stack(masks, axis=0).astype(int))

        # random crop
        if self.crop:
            if not self.test:

                if self.multi_scale is not None:
                    scale_factor = np.random.uniform(*self.multi_scale)
                    orig_shape = image.shape[-2:]
                    new_shape = (scale_factor * np.asarray(orig_shape)).astype(int).tolist()
                    image, mask = T.Resize(new_shape)(image), T.Resize(new_shape, interpolation=0)(mask)

                i, j, h, w = T.RandomCrop.get_params(image, self.crop)
                image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
            else:
                image, mask = T.CenterCrop(self.crop[0])(image), T.CenterCrop(self.crop[0])(mask)

        if self.p_flip and not self.test and np.random.rand() < self.p_flip:
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
        joint_transform: augmentation transform, an instance of JointTransform3D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
    """

    def __init__(self, dataset_path: str or List[str], modality_roi_map: List[dict], class_labels: List[str],
                 num_slices: int = None, slice_shape: Tuple[int] = None, crop_size: Tuple[int] = None,
                 joint_transform: Callable = None, patient_keys: List[str] or List[int] = None, num_patients: int = None,
                 patch_size: Tuple[int] = None, patch_stride: Tuple[int] = None,
                 patch_halo: Tuple[int] = None, **kwargs) -> None:
        # convert to a simple dict
        self.modality_roi_map = {list(item.keys())[0]: list(item.values())[0] for item in modality_roi_map}
        self.modality = list(self.modality_roi_map.keys())
        # useful inverse mapping which maps roi to modality
        self.roi_modality_map = {roi: m for m, r in self.modality_roi_map.items() for roi in r}
        self.class_labels = class_labels  # specifies the ordering of the channels (rois) in the mask array

        assert len(class_labels) > 0

        self.patient_keys = patient_keys  # this can either be a list of strings for keys or list of ints for indices
        self.num_slices = num_slices
        self.slice_shape = slice_shape
        self.crop_size = crop_size
        self.num_patients = num_patients  # used for train-val-test split
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_halo = patch_halo
        self.attend_samples = kwargs.get('attend_samples', False)
        self.attend_samples_all_axes = kwargs.get('attend_samples_all_axes', False)
        self.mask_samples = kwargs.get('mask_samples', False)
        self.frame_dict_path = kwargs.get('attend_frame_dict_path', None)
        self.logger = logging.getLogger(__name__)

        if self.slice_shape is None and len(self.modality) > 1:
            self.logger.warning("Doing multi channel but 'slice_shape' is set to None! "
                                "Use 'slice_shape' to specify a common resolution across modalities")

        if type(dataset_path) is str:
            self.dataset_path = [dataset_path]
        else:
            self.dataset_path = dataset_path

        self.dataset_dict = {}

        # handle case if multiple dataset paths passed
        for dp in self.dataset_path:
            with open(os.path.join(dp, "global_dict.json")) as file_obj:
                self.dataset_dict = {**self.dataset_dict, **json.load(file_obj)}

        if self.patient_keys is None:
            # if no patients specified then select all from dataset
            self.patient_keys = list(self.dataset_dict.keys())
        elif type(self.patient_keys[0]) is int:
            # if patient keys param is a list of indices get the keys from dataset
            all_patients = list(self.dataset_dict.keys())
            self.patient_keys = [all_patients[idx] for idx in self.patient_keys]

        # sample select patients if num_patients specified
        if num_patients is not None:
            selected_patients = np.random.choice(sorted(self.patient_keys), size=self.num_patients, replace=False)
            # keep track of excluded patients from selected
            self.excluded_patients = list(set(self.patient_keys) - set(selected_patients))
            self.patient_keys = selected_patients
        else:
            self.excluded_patients = list(set(self.dataset_dict.keys()) - set(self.patient_keys))

        self.all_patient_fps = {
            patient: {
                modality: './data/' + self.dataset_dict[patient][modality]["fp"]
                for modality in self.modality
            } for patient in self.patient_keys
        }

        if joint_transform is None:
            joint_transform = JointTransform3D(crop=None, p_flip=0, deform_sigma=None, div_by_max=False)
        self.joint_transform = joint_transform

        # if dataset is used during evaluation/testing then disable patch-wise during data fetching
        self.patch_wise = not self.joint_transform.test

        if self.patch_wise and self.patch_size is not None:
            dummy_img = torch.ones((1, self.num_slices, self.joint_transform.crop[0], self.joint_transform.crop[1]))
            dummy_msk = torch.ones(
                (len(self.class_labels), self.num_slices, self.joint_transform.crop[0], self.joint_transform.crop[1])
            )
            self.slicer = SliceBuilder([dummy_img], [dummy_msk], self.patch_size, self.patch_stride, None)
        else:
            self.slicer = None

    def __len__(self) -> int:
        if self.patch_wise and self.patch_size is not None:
            return len(self.patient_keys) * len(self.slicer.raw_slices)
        else:
            return len(self.patient_keys)

    def __getitem__(self, idx) -> dict:
        # divide index by number of patches to get patient idx
        if self.patch_wise and self.patch_size is not None:
            patient = list(self.patient_keys)[idx // len(self.slicer.raw_slices)]
        else:
            patient = list(self.patient_keys)[idx]

        patient_dir = self.all_patient_fps[patient]

        # read image
        image_dict = {
            modality: read_scan_as_sitk_image(patient_dir[modality]) for modality in self.modality
        }

        # get the raw image size for each modality
        raw_image_size_dict = {modality: image_dict[modality].GetSize()[::-1] for modality in self.modality}

        # TODO: get SUV values from PET
        if "CT" in self.modality:
            image_dict["CT"] = clamp_image_values(image=image_dict["CT"], lower_bound=-150, upper_bound=150)

        # read mask data from dataset and return mask dict
        mask_dict_npy = self.get_mask(patient, raw_image_size_dict)

        # perform resampling if multi channel/modality is specified
        if {*self.modality} == {"PT", "CT"} and self.slice_shape is not None:
            reference_size = [*self.slice_shape,
                              raw_image_size_dict["CT"][0]]  # assumes PET and CT have same image spacing in z direction
            image = combine_pet_ct_image(pet_image=image_dict["PT"],
                                         ct_image=downsample_image(image_dict["CT"], reference_size))

            mask_dict = {
                roi: mask_to_sitk_image(mask_dict_npy[roi], image_dict[self.roi_modality_map[roi]])
                for roi in mask_dict_npy
            }

            for roi in mask_dict:
                mask_dict[roi] = resample_image(mask_dict[roi], reference_image=image)
                # convert to npy
                mask_dict_npy[roi] = convert_image_to_npy(mask_dict[roi])

            # convert image to npy and set channels as first dimension
            image = np.moveaxis(
                convert_image_to_npy(image), source=-1, destination=0
            )

        else:
            # convert to npy and keep specified slices
            image_dict_npy = {
                modality: convert_image_to_npy(image_dict[modality]) for modality in self.modality
            }
            # generate a single ndarray
            image = np.asarray([*image_dict_npy.values()])

        # generate a single ndarray
        mask = np.asarray([*mask_dict_npy.values()])

        # need to have same tensor shape across samples in batch
        image, mask = image[:, :self.num_slices], mask[:, :self.num_slices]

        if self.crop_size is not None:
            image = centre_crop(image, (*image.shape[:2], *self.crop_size))
            mask = centre_crop(mask, (*mask.shape[:2], *self.crop_size))

        # keep copy of image before further image preprocessing
        orig_image = np.copy(image)

        if self.patch_wise and self.patch_size is not None:
            patch_idx = idx % len(self.slicer.raw_slices)
            patch_im_slice = self.slicer.raw_slices[patch_idx]
            patch_lab_slice = self.slicer.label_slices[patch_idx]
            image, mask = image[patch_im_slice], mask[patch_lab_slice]

        # reduce the search space for finding tumor
        if self.attend_samples:
            start_frame, end_frame = self.process_attend_indices(mask=mask, axes=(1,2))
            mask = mask[:, start_frame:end_frame, ...]
            image = image[:, start_frame:end_frame, ...]

        elif self.attend_samples_all_axes:
            depth_bounds = self.process_attend_indices(mask=mask, axes=(1,2))
            width_bounds = self.process_attend_indices(mask=mask, axes=(0,2))
            height_bounds = self.process_attend_indices(mask=mask, axes=(0,1))

            slices = tuple([slice(None)] + [slice(*i) for i in [depth_bounds, width_bounds, height_bounds]])

            mask = mask[slices]
            image = image[slices]

        elif self.mask_samples:
            depth_bounds = self.process_attend_indices(mask=mask, axes=(1,2))
            width_bounds = self.process_attend_indices(mask=mask, axes=(0,2))
            height_bounds = self.process_attend_indices(mask=mask, axes=(0,1))

            slices = tuple([slice(None)] + [slice(*i) for i in [depth_bounds, width_bounds, height_bounds]])

            mask_cp = np.zeros_like(mask)
            image_cp = np.zeros_like(image)

            mask_cp[slices] = mask[slices]
            mask = mask_cp

            image_cp[slices] = image[slices]
            image = image_cp

        # apply transforms and convert to tensors
        image, mask = self.joint_transform(image, mask)

        # compute distance map for boundary loss
        dist_map = one_hot2dist(np.asarray(mask), (1, 1, 1))

        return {
            "orig_image": orig_image,
            "image": image.float(),
            "gt_mask": mask.float(),
            "dist_map": dist_map,
            "patient": patient
        }

    def get_mask(self, patient: str, image_size_dict: Dict[str, Tuple]) -> dict:
        # get all rois from the dataset
        patient_rois = {
            modality: list(self.dataset_dict[patient][modality]["rois"].keys()) for modality in self.modality
        }

        # get specified roi data from dataset
        roi_data = {}
        for roi_name, modality in self.roi_modality_map.items():
            # check to see if specified rois exist in dataset
            if roi_name in patient_rois[modality]:
                roi_data[(roi_name, modality)] = self.dataset_dict[patient][modality]["rois"][roi_name]
            else:
                # add empty mask if specified roi does not exist in dataset
                # self.logger.warning("Roi '{}' does not exist in dataset for patient '{}'! Adding empty mask..."
                #                    .format(roi_name, patient))
                # creates an empty list of contours for each frame
                roi_data[(roi_name, modality)] = [[] for _ in range(image_size_dict[modality][0])]

        # build mask object for each roi
        mask = {
            roi_name: np.asarray(
                [contour2mask(roi_data[(roi_name, modality)][frame], image_size_dict[modality][1:])
                 for frame in range(image_size_dict[modality][0])]
            ) for roi_name, modality in roi_data
        }

        if "Tumor" in self.roi_modality_map:
            # call helper function to process tumor mask
            self.process_tumor_mask(mask)
        else:
            # add logic here for other combinations of rois
            pass

        # return properly ordered mask based on class labels (while ignoring background)
        return OrderedDict({
            roi_name: mask[roi_name] for roi_name in self.class_labels if roi_name != "Background"
        })

    def process_tumor_mask(self, mask: Dict[str, np.ndarray]) -> None:
        tumor_keys = [x for x in mask if "Tumor" in x]
        tumor_mask = np.zeros_like(mask["Tumor"])

        # merge Tumor rois into a single channel
        for tum in tumor_keys:
            tumor_mask += mask[tum]
            if tum != "Tumor":
                del mask[tum]

        tumor_mask[tumor_mask > 1] = 1

        if "Bladder" in self.roi_modality_map:
            mask["Bladder"][tumor_mask == 1] = 0  # ensure there is no overlap in gt bladder mask

        # update tumor mask in the mask dict
        mask["Tumor"] = tumor_mask
    
    def process_attend_indices(self, mask, axes):
        mins = []
        maxs = []
        for roi in ['Bladder', 'Prostate']:
            roi_idx = self.class_labels.index(roi)
            bounds = mask[roi_idx].sum(axis=axes)
            bounds = np.nonzero(bounds)
            mins.append(bounds[0])
            maxs.append(bounds[-1])

        return (min(mins), max(maxs))



class Image3D(Dataset):
    """
    Dataset class purely for inference TODO: test me
    """

    def __init__(self, dataset_path: str or List[str], path_suffix: str = "", transform: Callable = None,
                 **kwargs) -> None:
        if type(dataset_path) is str:
            self.dataset_path = [dataset_path]
        else:
            self.dataset_path = dataset_path

        # get all the subdirectories containing scans
        self.scan_fps = [
            os.path.join(scan_path, path_suffix) for dp in self.dataset_path
            for scan_path in glob.glob(os.path.join(dp, "/*/"))
        ]

        self.transform = transform or torch.Tensor

    def __len__(self) -> int:
        return len(self.scan_fps)

    def __getitem__(self, idx):
        scan = list(self.scan_fps)[idx]

        # read image
        image = read_scan_as_sitk_image(scan)

        # convert to npy array
        image = convert_image_to_npy(image)

        return self.transform(image)
