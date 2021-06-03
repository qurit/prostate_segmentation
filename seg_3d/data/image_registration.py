from typing import Tuple

import SimpleITK as sitk
import numpy as np


def read_scan_as_sitk_image(dcm_dir: str) -> sitk.Image:
    series_reader = sitk.ImageSeriesReader()
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_dir)

    # get all the .dcm files from directory
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir, series_IDs[0])
    series_reader.SetFileNames(series_file_names)

    # return scan
    return series_reader.Execute()


def resample_image(image: sitk.Image, reference_image: sitk.Image = None, size: Tuple[int] = None) -> sitk.Image:
    assert reference_image is not None or size is not None, "cannot have both args as None"

    if reference_image is not None:
        return sitk.Resample(image, referenceImage=reference_image)

    if len(size) == 2:
        size = (*size, image.GetDepth())

    return sitk.Resample(image, size=sitk.VectorUInt32(size))


def convert_image_to_npy(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def mask_to_sitk_image(mask: np.ndarray, image: sitk.Image) -> sitk.Image:
    mask = sitk.GetImageFromArray(mask)
    mask.CopyInformation(image)
    return mask
