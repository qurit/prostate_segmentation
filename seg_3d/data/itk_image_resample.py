from typing import List

import SimpleITK as sitk
import numpy as np

__all__ = [
    'print_image_info',
    'read_scan_as_sitk_image',
    'downsample_image',
    'combine_pet_ct_image',
    'convert_image_to_npy',
    'resample_image',
    'mask_to_sitk_image',
    'clamp_image_values'
]


def print_image_info(image: sitk.Image, name=''):
    print('information', name)
    print('\timage size: {0}'.format(image.GetSize()))
    print('\timage spacing: {0}'.format(image.GetSpacing()))
    print('\tpixel type: ' + image.GetPixelIDTypeAsString())
    print('\tnumber of channels: ' + str(image.GetNumberOfComponentsPerPixel()))


def read_scan_as_sitk_image(dcm_dir: str) -> sitk.Image:
    """ Thanks to https://discourse.itk.org/t/compose-image-from-different-modality-with-different-number-of-slices/2286/8 """
    series_reader = sitk.ImageSeriesReader()
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_dir)

    # get all the .dcm files from directory
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir, series_IDs[0])
    series_reader.SetFileNames(series_file_names)

    # print dicom metadata
    # import pydicom
    # print(pydicom.filereader.dcmread(series_file_names[0]))

    # return scan
    return series_reader.Execute()


def downsample_image(original_image: sitk.Image, reference_size: List) -> sitk.Image:
    """ Thanks to https://stackoverflow.com/a/63120034 """
    dimension = original_image.GetDimension()
    reference_physical_size = np.zeros(original_image.GetDimension())
    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(original_image.GetSize(), original_image.GetSpacing(), reference_physical_size)]

    reference_origin = original_image.GetOrigin()
    reference_direction = original_image.GetDirection()

    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, original_image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(original_image.GetDirection())

    transform.SetTranslation(np.array(original_image.GetOrigin()) - reference_origin)

    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(original_image.TransformContinuousIndexToPhysicalPoint(np.array(original_image.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.CompositeTransform(transform)
    centered_transform.AddTransform(centering_transform)

    return sitk.Resample(original_image, reference_image, centered_transform, sitk.sitkLinear, 0.0)


def combine_pet_ct_image(pet_image: sitk.Image, ct_image: sitk.Image, verbose=False) -> sitk.Image:
    # Resample PET onto CT grid using default interpolator and identity transformation.
    pet_image_resampled = sitk.Resample(pet_image, ct_image)

    # Compose the PET and CT image into a single two channel image.
    # The pixel types of all channels need to match, so we upcast the CT from
    # 32bit signed int to the PET pixel type of 64bit float.
    pet_ct_combined = sitk.Compose(pet_image_resampled, sitk.Cast(ct_image, pet_image_resampled.GetPixelID()))

    if verbose:
        for image, image_name in zip([pet_image, ct_image, pet_ct_combined], ['PET', 'CT', 'Combined PET-CT']):
            print_image_info(image, image_name)

    return pet_ct_combined


def convert_image_to_npy(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def resample_image(image: sitk.Image, reference_image: sitk.Image, interpolator=sitk.sitkNearestNeighbor) -> sitk.Image:
    return sitk.Resample(image, reference_image, interpolator=interpolator)


def mask_to_sitk_image(mask: np.ndarray, image: sitk.Image) -> sitk.Image:
    mask = sitk.GetImageFromArray(mask)
    mask.CopyInformation(image)
    return mask


def clamp_image_values(image: sitk.Image, lower_bound: int, upper_bound: int) -> sitk.Image:
    clamp_filter = sitk.ClampImageFilter()
    clamp_filter.SetLowerBound(lower_bound)
    clamp_filter.SetUpperBound(upper_bound)
    return clamp_filter.Execute(image)
