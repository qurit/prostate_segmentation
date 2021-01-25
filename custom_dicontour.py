from contour_utils import *
from collections import defaultdict
import os
import operator
import warnings


def get_contour_file(path):
    """
    Get contour file from a given path by searching for ROIContourSequence
    inside dicom data structure.
    More information on ROIContourSequence available here:
    http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html

    Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
    Return:
        contour_file (str): name of the file with the contour
    """
    # get .dcm contour file
    fpaths = [os.path.join(path, f) for f in os.listdir(path) if '.dcm' in f]
    n = 0
    contour_file, contour_path = None, None
    for fpath in fpaths:
        f = dicom.read_file(fpath)
        if 'ROIContourSequence' in dir(f):
            contour_path = fpath
            contour_file = fpath.split('/')[-1]
            n += 1
    if n > 1:
        warnings.warn("There are multiple contour files, returning the last one!")
    if contour_file is None:
        print("No contour file found in directory")
    return contour_path, contour_file


def coord2pixels(contour_dataset, path):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images. This function will return img_arr and contour_arr (2d image and contour pixels)
    Inputs
        contour_dataset: DICOM dataset class that is identified as (3006, 0016)  Contour Image Sequence
        path: string that tells the path of all DICOM images
    Return
        img_arr: 2d np.array of image with pixel intensities
        contour_arr: 2d np.array of contour with 0 and 1 labels
    """

    contour_coord = contour_dataset.ContourData
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    # extract the image id corresponding to given countour
    # read that dicom file
    img_id = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    img = dicom.read_file(os.path.join(path, img_id + '.dcm'))

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.ceil((x - origin_x) / x_spacing), np.ceil((y - origin_y) / y_spacing)) for x, y, _ in coord]

    return pixel_coords, img_id


def slice_order(path):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """

    slices = []
    for s in os.listdir(path):
        f = dicom.read_file(os.path.join(path, s))
        if f.Modality != 'RTSTRUCT':
            slices.append(f)

    slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
    return ordered_slices


def get_contour_dict(contour_file, path, index):
    """
    Returns a dictionary as k: img fname, v: [corresponding img_arr, corresponding contour_arr]
    Inputs:
        contour_file: .dcm contour file name
        path: path which has contour and image files
    Returns:
        contour_dict: dictionary with 2d np.arrays
    """
    f = dicom.read_file(os.path.join(path, contour_file))
    roi = f.ROIContourSequence[index]

    # get contour datasets in a list
    contours = [contour for contour in roi.ContourSequence]
    img_contour_arrays = [coord2pixels(cdata, path) for cdata in contours]  # list of img_arr, contour_arr, im_id

    # debug: there are multiple contours for the same image independently
    # append contour arrays and generate new img_contour_arrays
    contour_dict = defaultdict(list)
    for cntr_arr, im_id in img_contour_arrays:
        contour_dict[im_id].append(cntr_arr)

    return contour_dict


def get_data(path):
    """
    Generate image array and contour array
    Inputs:
        path (str): path of the the directory that has DICOM files in it
        contour_dict (dict): dictionary created by get_contour_dict
        index (int): index of the
    """
    images = []

    # get contour file
    contour_path, contour_file = get_contour_file(path)
    # get slice orders
    ordered_slices = slice_order(path)

    patient_id, modality, position = None, None, None

    # get image array
    for i, (k, v) in enumerate(ordered_slices):
        dcm = dicom.read_file(os.path.join(path, k + '.dcm'))
        img_arr = parse_dicom_image(dcm)
        images.append(img_arr)
        if i == 0:
            patient_id = dcm.PatientID
            modality = dcm.Modality
            position = dcm.PatientPosition

    assert None not in [patient_id, modality, position]

    # get roi names
    roi_list = [x.ROIName for x in dicom.read_file(contour_path).StructureSetROISequence]

    # get contour dict
    contour_dict = {}
    for index, roi in enumerate(roi_list):
        contour_dict[roi] = get_contour_dict(contour_file, path, index)

    contours = {'patientID': patient_id, 'modality': modality, 'position': position}
    for roi, cdict in contour_dict.items():
        contours[roi] = []
        for index, (k, v) in enumerate(ordered_slices):
            # get data from contour dict
            if k in cdict:
                contours[roi].append(cdict[k])
            # get data from dicom.read_file
            else:
                contours[roi].append([])

    return patient_id, modality, np.array(images), contours
