"""Original code taken from https://github.com/KeremTurgutlu/dicom-contour"""
import os

import scipy
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_modality_lut


def get_smallest_dcm(path, ext='.dcm'):
    """
    Get smallest dcm file in size given path of target dir

    Args:
        path (str): path of the the directory that has DICOM files in it
        ext (str): extension of the DICOM files are defined with
    """
    fsize_dict = {f: os.path.getsize(path + f) for f in os.listdir(path)}
    for fname, size in [(k, fsize_dict[k]) for k in sorted(fsize_dict, key=fsize_dict.get, reverse=False)]:
        if ext in fname:
            return fname


def get_roi_names(contour_data):
    """
    This function will return the names of different contour data,
    e.g. different contours from different experts and returns the name of each.

    Args:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file

    Returns:
        roi_seq_names (list)
    """
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names


def parse_dicom_image(dcm):
    """Parse the given DICOM filename
    :param dcm: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """
    try:
        dcm_image = dcm.pixel_array.astype(np.float)
        dcm_image = apply_modality_lut(dcm_image, dcm)
        return dcm_image
    except dicom.errors.InvalidDicomError:
        return None

def coords2poly(contour_dataset, path):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images return polygon coordinates for the contours.

    Args:
        contour_dataset (dicom.dataset.Dataset) : DICOM dataset class that is identified as
                         (3006, 0016)  Contour Image Sequence
        path (str): path of directory containing DICOM images

    Return:
        pixel_coords (list): list of tuples having pixel coordinates
        img_ID (id): DICOM image id which maps input contour dataset
        img_shape (tuple): DICOM image shape - height, width
    """
    contour_coord = contour_dataset.ContourData
    # x, y, z coordinates of the contour in mm
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    # extract the image id corresponding to given countour
    # read that dicom file
    img_id = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    img = dicom.read_file(path + img_id + '.dcm')
    img_arr = img.pixel_array
    img_shape = img_arr.shape

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.ceil((x - origin_x) / x_spacing), np.ceil((y - origin_y) / y_spacing)) for x, y, _ in coord]
    return pixel_coords, img_id, img_shape


def poly2contour(contour_data, shape):
    # get contour data for the image
    rows = []
    cols = []
    for i, j in list(set(contour_data)):
        rows.append(i)
        cols.append(j)
    contour_arr = scipy.sparse.csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8,
                                          shape=(shape[0], shape[1])).toarray()
    return contour_arr


def contour2mask(contours, size, xy_ordering=True):
    """
    Converts contour data to a binary mask.
    Binary masks are easier to work with when it comes to
    learning segmentation models but are resource intensive
    
    Args:
    contours (list): contour data where each contour is a list of shape Nx2
    size (Tuple): size of the original image

    Return:
        mask (np.ndarray): binary mask
    """
    mask = np.zeros(size, int)
    if len(contours) == 0:
        # handle case for empty contours, return empty mask in that case
        return mask

    poly_set = [np.asarray(poly) for poly in contours]
    for poly in poly_set:
        if xy_ordering:
            # case for polygon coordinates being (x,y)
            mask[poly[:, 1], poly[:, 0]] = 1
        else:
            # case for polygon coordinates being (row, column)
            mask[poly[:, 0], poly[:, 1]] = 1
    mask = ndimage.morphology.binary_fill_holes(mask)
    return mask


def plot2dcontour(img_arr, contour_arr, figsize=(20, 20)):
    """
    Shows 2d img with contour

    Args:
        img_arr: 2d np.array image array with pixel intensities
        contour_arr: 2d np.array contour array with pixels of 1 and 0
    """
    masked_contour_arr = np.ma.masked_where(contour_arr == 0, contour_arr)
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr, cmap='gray', interpolation='none')
    plt.subplot(1, 2, 2)
    plt.imshow(img_arr, cmap='gray', interpolation='none')
    plt.imshow(masked_contour_arr, cmap='cool', interpolation='none', alpha=0.7)
    plt.show()
