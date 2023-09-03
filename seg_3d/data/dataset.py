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
from seg_3d.utils.misc_utils import one_hot2dist, centre_crop
from dicom_code.contour_utils import contour2mask


# global vars
CONTOUR_DATA_DICT_FILE = 'global_dict.json'
ROOT_DATA_DIR = './data/'


class JointTransform3D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        test: set to True if in test mode which disables some of the augmentations and randomness
        crop_size: tuple describing the size of the random crop. If bool(crop_size) evaluates to False, no crop will be taken.
        p_flip: the probability of performing a random horizontal flip.
        deform_sigma: sigma parameter used for elastic deformation
        deform_points: points parameter used for elastic deformation
        multi_scale: min and max scaling factors for random scaling, e.g. multi_scale=(0.85,1.15)
        ignore_bg: if True, then no mask is generated for the background class
    """

    def __init__(self, test: bool = False, crop_size: Tuple = None, p_flip: float = None, deform_sigma: float = None,
                 deform_points: Tuple or int = (3, 3, 3), min_max_norm: bool = True, multi_scale: Tuple = None,
                 ignore_bg: bool = False):
        self.test = test
        self.crop_size = crop_size
        self.p_flip = p_flip
        self.min_max_norm = min_max_norm
        self.multi_scale = multi_scale
        self.ignore_bg = ignore_bg

        if deform_sigma and not self.test:
            self.deform = lambda x, y: \
                elasticdeform.deform_random_grid([*x, *y], sigma=deform_sigma, points=deform_points,
                                                 order=[*[3] * len(x),
                                                        *[0] * len(y)])  # order must be 0 for mask arrays
        else:
            self.deform = lambda x, y: [*x, *y]

    def __call__(self, image: np.ndarray, masks: np.ndarray) -> Tuple[torch.tensor, torch.tensor]:

        # normalization
        # feature scaling https://en.wikipedia.org/wiki/Feature_scaling
        if self.min_max_norm:
            image = np.asarray([(im - im.min()) / np.ptp(im) for im in image])  # dividing by max - min is bad if it equals 0...
            assert (0 <= image.min() < image.max() <= 1), 'Input data is not [0,1] normalized!!'

        # get number of channels in image
        img_channels = len(image)

        # apply elastic deformation if specified
        sample_data = self.deform(image, masks)
        image, masks = sample_data[0:img_channels], sample_data[img_channels:]

        # this step needs to happen after elastic deformation
        if not self.ignore_bg:
            bg = np.ones_like(masks[0])
            for m in masks:
                bg[bg == m] = 0
            masks = [bg, *masks]  # the first entry in masks is background
        
        # transforming to tensor
        image = torch.Tensor(np.array(image))
        mask = torch.Tensor(np.stack(masks, axis=0).astype(int))

        # random crop
        if self.crop_size:
            if not self.test:

                if self.multi_scale is not None:
                    scale_factor = np.random.uniform(*self.multi_scale)
                    orig_shape = image.shape[-2:]
                    new_shape = (scale_factor * np.asarray(orig_shape)).astype(int).tolist()
                    image, mask = T.Resize(new_shape)(image), T.Resize(new_shape, interpolation=0)(mask)

                i, j, h, w = T.RandomCrop.get_params(image, self.crop_size)
                image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
            else:
                image, mask = T.CenterCrop(self.crop_size[0])(image), T.CenterCrop(self.crop_size[0])(mask)

        if self.p_flip and not self.test and np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        return image, mask


class ImageToImage3D(Dataset):
    """
    Custom Dataset class for pytorch pipeline. Assumes dataset is formatted in custom format
    and is stored inside REPO_ROOT_DIR/data/

    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- global_dict.json  // contains metadata and contour data for the scans of each patient
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
        modality_roi_map: a dict for each modality to specify the ROIs and modalities to extract from dataset
            e.g. modality_roi_map=[{'PT': ['Bladder', 'Tumor']}]
        class_labels: specifies which ROIs to include in the final mask Tensor as well as the ordering
            e.g. class_labels=['Background', 'Bladder', 'Tumor']
        num_slices: number of axial slices to include
        slice_shape: the dimensions to configure an axial slice, needed for multi-modality training!
        crop_size: tuple describing the size of the initial crop
        clamp_ct: tuple describing the lower and upper bounds for image clamping of CT scan
        joint_transform: an instance of JointTransform3D for data augmentations. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        modality: specifies modality of scan
        rois: list of region of interests
        patient_keys: specifies patients to include either by index or patient ID, if patient_keys=None
            then all patients from dataset are loaded
        num_patients: specifies the number of patients to include, sampled randomly
    """

    def __init__(self, dataset_path: str or List[str], modality_roi_map: List[dict], class_labels: List[str],
                 num_slices: int = None, slice_shape: Tuple[int] = None, crop_size: Tuple[int] = None, clamp_ct: Tuple[int] = (-150, 150),
                 joint_transform: Callable = None, patient_keys: List[str] or List[int] = None, num_patients: int = None):
        # convert to a simple dict
        self.modality_roi_map = {list(item.keys())[0]: list(item.values())[0] for item in modality_roi_map}
        self.modality = list(self.modality_roi_map.keys())
        # useful inverse mapping which maps roi to modality
        self.roi_modality_map = {roi: m for m, r in self.modality_roi_map.items() for roi in r}
        self.class_labels = class_labels  # specifies the ordering of the channels (rois) in the mask tensor
        assert len(class_labels) > 0
        self.patient_keys = patient_keys  # this can either be a list of strings for keys or list of ints for indices
        self.num_slices = num_slices
        self.slice_shape = slice_shape
        self.crop_size = crop_size
        self.clamp_ct = clamp_ct
        self.num_patients = num_patients  # can be used for train-val-test split
        self.logger = logging.getLogger(__name__)

        if self.slice_shape is None and len(self.modality) > 1:
            self.logger.warning('Doing multi channel but "slice_shape" is set to None! '
                                'Use "slice_shape" to specify a common resolution across modalities')

        if type(dataset_path) is str:
            self.dataset_path = [dataset_path]
        else:
            self.dataset_path = dataset_path

        self.dataset_dict = {}

        # handle case if multiple dataset paths passed
        for dp in self.dataset_path:
            with open(os.path.join(dp, CONTOUR_DATA_DICT_FILE)) as file_obj:
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
                modality: ROOT_DATA_DIR + self.dataset_dict[patient][modality]['fp']
                for modality in self.modality
            } for patient in self.patient_keys
        }

        if joint_transform is None:
            joint_transform = JointTransform3D(crop_size=None, p_flip=0, deform_sigma=None, min_max_norm=False)
        self.joint_transform = joint_transform

    def get_patient_by_name(self, name) -> dict:
        for idx, item in enumerate(self.patient_keys):
            if item == name:
                return self[idx]

    def __len__(self) -> int:
            return len(self.patient_keys)

    def __getitem__(self, idx) -> dict:
        patient = list(self.patient_keys)[idx]
        patient_dir = self.all_patient_fps[patient]

        # read image
        image_dict = {
            modality: read_scan_as_sitk_image(patient_dir[modality]) for modality in self.modality
        }

        # get the raw image size for each modality
        raw_image_size_dict = {modality: image_dict[modality].GetSize()[::-1] for modality in self.modality}

        # preprocessing of CT
        if 'CT' in self.modality and bool(self.clamp_ct):
            image_dict['CT'] = clamp_image_values(image=image_dict['CT'], lower_bound=self.clamp_ct[0], upper_bound=self.clamp_ct[1])

        # preprocessing of PET
        if 'PT' in self.modality:
            pass

        # read mask data from dataset and return mask dict
        mask_dict_npy = self.get_mask(patient, raw_image_size_dict)

        # perform resampling if multi channel/modality is specified
        if {*self.modality} == {'PT', 'CT'} and self.slice_shape is not None:
            reference_size = [*self.slice_shape,
                              raw_image_size_dict['CT'][0]]  # assumes PET and CT have same image spacing in z direction
            image = combine_pet_ct_image(pet_image=image_dict['PT'],
                                         ct_image=downsample_image(image_dict['CT'], reference_size), verbose=False)

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

        # keep copy of image before further image preprocessing
        orig_image = np.copy(image)

        if self.crop_size is not None:
            image = centre_crop(image, (*image.shape[:2], *self.crop_size))
            mask = centre_crop(mask, (*mask.shape[:2], *self.crop_size))

        # apply transforms and convert to tensors
        image, mask = self.joint_transform(image, mask)

        # compute distance map for boundary loss
        dist_map = one_hot2dist(np.asarray(mask), (1, 1, 1))

        return {
            'orig_image': orig_image,
            'image': image.float(),
            'gt_mask': mask.float(),
            'dist_map': dist_map,  # used for boundary loss
            'patient': patient  # patient id
        }

    def get_mask(self, patient: str, image_size_dict: Dict[str, Tuple]) -> dict:
        # get all rois from the dataset
        patient_rois = {
            modality: list(self.dataset_dict[patient][modality]['rois'].keys()) for modality in self.modality
        }

        # get specified roi data from dataset
        roi_data = {}
        for roi_name, modality in self.roi_modality_map.items():
            # check to see if specified rois exist in dataset
            if roi_name in patient_rois[modality]:
                roi_data[(roi_name, modality)] = self.dataset_dict[patient][modality]['rois'][roi_name]
            else:
                # add empty mask if specified roi does not exist in dataset
                # self.logger.warning('Roi "{}" does not exist in dataset for patient "{}"! Adding empty mask...'
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

        if 'Tumor' in self.roi_modality_map:
            # call helper function to process tumor mask
            mask = self.process_tumor_mask(mask)
        else:
            # add logic here for other combinations of rois
            pass

        # return properly ordered mask based on class labels (while ignoring background)
        return OrderedDict({
            roi_name: mask[roi_name] for roi_name in self.class_labels if roi_name != 'Background'
        })

    def process_tumor_mask(self, mask: Dict[str, np.ndarray]) -> None:
        """
        Merges all the tumor masks into a single class called 'Tumor'
        Some patients may have 'Tumor', 'Tumor2', and 'Tumor3'
        """
        tumor_keys = [x for x in mask if 'Tumor' in x]
        tumor_mask = np.zeros_like(mask['Tumor'])

        # merge Tumor rois into a single channel
        for tum in tumor_keys:
            tumor_mask += mask[tum]
            if tum != 'Tumor':
                del mask[tum]

        tumor_mask[tumor_mask > 1] = 1

        if 'Bladder' in self.roi_modality_map:
            mask['Bladder'][tumor_mask == 1] = 0  # ensure there is no overlap in gt bladder mask

        # update tumor mask in the mask dict
        mask['Tumor'] = tumor_mask

        return mask

class ImageToImage3DInference(Dataset):

    def __init__(self, dataset_path: str, modality_roi_map: List[dict], class_labels: List[str],
                 num_slices: int = None, slice_shape: Tuple[int] = None, crop_size: Tuple[int] = None, clamp_ct: Tuple[int] = (-150, 150),
                 joint_transform: Callable = None, patient_keys: List[str] or List[int] = None, num_patients: int = None):
        # convert to a simple dict
        self.dataset_path = dataset_path
        self.modality_roi_map = {list(item.keys())[0]: list(item.values())[0] for item in modality_roi_map}
        self.modality = list(self.modality_roi_map.keys())
        # useful inverse mapping which maps roi to modality
        self.roi_modality_map = {roi: m for m, r in self.modality_roi_map.items() for roi in r}
        self.class_labels = class_labels  # specifies the ordering of the channels (rois) in the mask tensor
        assert len(class_labels) > 0
        self.num_slices = num_slices
        self.slice_shape = slice_shape
        self.crop_size = crop_size
        self.clamp_ct = clamp_ct
        self.num_patients = num_patients  # can be used for train-val-test split
        self.logger = logging.getLogger(__name__)

        if self.slice_shape is None and len(self.modality) > 1:
            self.logger.warning('Doing multi channel but "slice_shape" is set to None! '
                                'Use "slice_shape" to specify a common resolution across modalities')

        if joint_transform is None:
            joint_transform = JointTransform3D(crop_size=None, p_flip=0, deform_sigma=None, min_max_norm=False)
        self.joint_transform = joint_transform

        self.scan_keys = os.listdir(self.dataset_path)

    def __len__(self) -> int:
            return len(self.scan_keys)

    def __getitem__(self, idx) -> dict:
        scan = self.scan_keys[idx]
        scan_dir = os.path.join(self.dataset_path, scan)

        for modality in self.modalities:
            scan_dir = os.path.join(patient_dir, modality)
            assert os.path.exists

        # read image
        image_dict = {
            modality: read_scan_as_sitk_image(patient_dir[modality]) for modality in self.modality
        }

        # get the raw image size for each modality
        raw_image_size_dict = {modality: image_dict[modality].GetSize()[::-1] for modality in self.modality}

        # preprocessing of CT
        if 'CT' in self.modality and bool(self.clamp_ct):
            image_dict['CT'] = clamp_image_values(image=image_dict['CT'], lower_bound=self.clamp_ct[0], upper_bound=self.clamp_ct[1])

        # preprocessing of PET
        if 'PT' in self.modality:
            pass

        # read mask data from dataset and return mask dict
        mask_dict_npy = self.get_mask(patient, raw_image_size_dict)

        # perform resampling if multi channel/modality is specified
        if {*self.modality} == {'PT', 'CT'} and self.slice_shape is not None:
            reference_size = [*self.slice_shape,
                              raw_image_size_dict['CT'][0]]  # assumes PET and CT have same image spacing in z direction
            image = combine_pet_ct_image(pet_image=image_dict['PT'],
                                         ct_image=downsample_image(image_dict['CT'], reference_size), verbose=False)

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

        # keep copy of image before further image preprocessing
        orig_image = np.copy(image)

        if self.crop_size is not None:
            image = centre_crop(image, (*image.shape[:2], *self.crop_size))
            mask = centre_crop(mask, (*mask.shape[:2], *self.crop_size))

        # apply transforms and convert to tensors
        image, mask = self.joint_transform(image, mask)

        # compute distance map for boundary loss
        dist_map = one_hot2dist(np.asarray(mask), (1, 1, 1))

        return {
            'orig_image': orig_image,
            'image': image.float(),
            'gt_mask': mask.float(),
            'dist_map': dist_map,  # used for boundary loss
            'patient': patient  # patient id
        }

    def get_mask(self, patient: str, image_size_dict: Dict[str, Tuple]) -> dict:
        # get all rois from the dataset
        patient_rois = {
            modality: list(self.dataset_dict[patient][modality]['rois'].keys()) for modality in self.modality
        }

        # get specified roi data from dataset
        roi_data = {}
        for roi_name, modality in self.roi_modality_map.items():
            # check to see if specified rois exist in dataset
            if roi_name in patient_rois[modality]:
                roi_data[(roi_name, modality)] = self.dataset_dict[patient][modality]['rois'][roi_name]
            else:
                # add empty mask if specified roi does not exist in dataset
                # self.logger.warning('Roi "{}" does not exist in dataset for patient "{}"! Adding empty mask...'
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

        if 'Tumor' in self.roi_modality_map:
            # call helper function to process tumor mask
            mask = self.process_tumor_mask(mask)
        else:
            # add logic here for other combinations of rois
            pass

        # return properly ordered mask based on class labels (while ignoring background)
        return OrderedDict({
            roi_name: mask[roi_name] for roi_name in self.class_labels if roi_name != 'Background'
        })

    def process_tumor_mask(self, mask: Dict[str, np.ndarray]) -> None:
        """
        Merges all the tumor masks into a single class called 'Tumor'
        Some patients may have 'Tumor', 'Tumor2', and 'Tumor3'
        """
        tumor_keys = [x for x in mask if 'Tumor' in x]
        tumor_mask = np.zeros_like(mask['Tumor'])

        # merge Tumor rois into a single channel
        for tum in tumor_keys:
            tumor_mask += mask[tum]
            if tum != 'Tumor':
                del mask[tum]

        tumor_mask[tumor_mask > 1] = 1

        if 'Bladder' in self.roi_modality_map:
            mask['Bladder'][tumor_mask == 1] = 0  # ensure there is no overlap in gt bladder mask

        # update tumor mask in the mask dict
        mask['Tumor'] = tumor_mask

        return mask

class Image3D(Dataset):
    """
    Dataset class purely for inference

    TODO: this is NOT implemented! Needs the preprocessing (e.g. resampling 
    for multi modality, centre crop, etc.) of data minus label handling

    nice to haves:
     - option to run inference on a particular patient specified by id
     - returns prediction as a RTSTRUCT
    """

    def __init__(self, dataset_path: str or List[str], path_suffix: str = '', transform: Callable = None,
                 **kwargs) -> None:
        if type(dataset_path) is str:
            self.dataset_path = [dataset_path]
        else:
            self.dataset_path = dataset_path

        # get all the subdirectories containing scans
        self.scan_fps = [
            os.path.join(scan_path, path_suffix) for dp in self.dataset_path
            for scan_path in glob.glob(os.path.join(dp, '/*/'))
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
