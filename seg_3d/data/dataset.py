# original code from https://github.com/cosmic-cortex/pytorch-UNet/blob/master/unet/dataset.py
import glob
import json
import logging
import os
from typing import Callable
from typing import List, Tuple

import elasticdeform
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
import numpy as np
import pydicom as dicom
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

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

# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=0 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=50, sigma=1, execution_probability=1., apply_3d=True,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim in [3, 4]

            if m.ndim == 3:
                volume_shape = m.shape
            else:
                volume_shape = m[0].shape

            if self.apply_3d:
                dz = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="reflect") * self.alpha
            else:
                dz = np.zeros_like(m)

            dy, dx = [
                gaussian_filter(
                    self.random_state.randn(*volume_shape),
                    self.sigma, mode="reflect"
                ) * self.alpha for _ in range(2)
            ]

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx

            if m.ndim == 3:
                return map_coordinates(m, indices, order=self.spline_order, mode='reflect')
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
                return np.stack(channels, axis=0)

        return m




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

    def __init__(self, test=False, crop=(100, 100), p_flip=0.5, deform=None, div_by_max=True):

        self.crop = crop
        
        self.p_flip = p_flip
        
        if deform:
            # self.deform = lambda x: elasticdeform.deform_random_grid(x, sigma=deform)
            self.deform = ElasticDeformation(np.random.RandomState(seed=15), spline_order=0)
        else:
            self.deform = lambda x: x
            
        self.div_by_max = div_by_max

        self.test = test

    def __call__(self, image, masks):

        # divide by scan max
        if self.div_by_max:
            image = image / np.max(image)

        sample_data = self.deform(np.stack([image] + masks, axis=0))

        image, masks = sample_data[0], sample_data[1:].astype(int)

        # transforming to tensor
        image = torch.Tensor(image)
        mask = torch.Tensor(np.stack(masks, axis=0))
    
        # random crop
        if self.crop:
            if not self.test:
                i, j, h, w = T.RandomCrop.get_params(image, self.crop)
                image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
            else:
                image, mask = T.CenterCrop(self.crop[0])(image), T.CenterCrop(self.crop[0])(mask)

        if self.p_flip:
            if np.random.rand() < self.p_flip:
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
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, modality: str, rois: List[str], num_slices: int, crop_size: Tuple[int],
                 joint_transform: Callable, patient_keys: List[str] = None, one_hot_mask: int = False,
                 num_patients: int = None) -> None:
        self.dataset_path = dataset_path
        self.modality = modality
        self.patient_keys = patient_keys
        self.rois = rois
        self.num_slices = num_slices
        self.crop_size = crop_size
        self.one_hot_mask = one_hot_mask
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
            self.excluded_patients = list(set(self.patient_keys) - set(selected_patients))
            self.patient_keys = selected_patients

        self.all_frame_fps = {patient: glob.glob('data/' + self.dataset_dict[patient][self.modality]['fp'] + "/*.dcm")
                              for patient in self.patient_keys}

        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.patient_keys)

    def __getitem__(self, idx):
        patient = list(self.patient_keys)[idx]
        frame_fps = sorted(self.all_frame_fps[patient], key=lambda x: int(os.path.basename(x).split('.')[0]))

        # read image
        image = np.asarray([parse_dicom_image(dicom.dcmread(fp)) for fp in frame_fps][:self.num_slices]).astype(np.float32)
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

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, *mask.shape[1:])).scatter_(0, mask.long(), 1)

        return {
            "orig_image": orig_image,
            "image": image.unsqueeze(0).float(),
            "gt_mask": mask.float(),
            "patient": patient
        }

    def get_mask(self, patient, image):

        img_size = np.shape(image)

        patient_rois = self.dataset_dict[patient][self.modality]['rois'].keys()
        roi_data = [(roi, self.dataset_dict[patient][self.modality]['rois'][roi]) for roi in self.rois if roi in patient_rois]

        mask = {roi[0]: np.asarray(
            [contour2mask(roi[1][frame], img_size[1:3]) for frame in range(len(self.all_frame_fps[patient]))][:self.num_slices]
            ) for roi in roi_data}
        
        if 'Tumor' in self.rois:
            tumor_keys = [x for x in mask.keys() if 'Tumor' in x]
            tumor_mask = np.zeros_like(mask[tumor_keys[0]])
            for tum in tumor_keys:
                tumor_mask += mask[tum]
                del mask[tum]

            tumor_mask[tumor_mask > 1] = 1
            mask['Bladder'][tumor_mask == 1] = 0
        bg = np.zeros(tumor_mask.shape)
        bg = mask['Bladder'] + tumor_mask + 1
        bg[bg != 1] = 0
        masks_array = [bg, mask['Bladder'], tumor_mask]  # FIXME: hardcoded

        return masks_array


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
