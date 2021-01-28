"""Original code taken from https://github.com/KeremTurgutlu/dicom-contour"""
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
        contour_path (str): path of the file with the contour
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
    corresponding images return polygon coordinates for the contours.
    Inputs
        contour_dataset: DICOM dataset class that is identified as (3006, 0016)  Contour Image Sequence
        path: string that tells the path of all DICOM images
    Return
        pixel_coords (list): list of tuples having pixel coordinates
        img_ID (id): DICOM image id which maps input contour dataset
    """

    contour_coord = contour_dataset.ContourData
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((float(contour_coord[i]), float(contour_coord[i + 1])))

    # extract the image id corresponding to given contour
    # read that dicom file
    img_id = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    img = dicom.read_file(os.path.join(path, img_id + '.dcm'))

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y = float(img.ImagePositionPatient[0]), float(img.ImagePositionPatient[1])

    # y, x is how it's mapped
    pixel_coords = list(set([(int((x - origin_x) / x_spacing), int((y - origin_y) / y_spacing)) for x, y in coord]))

    return pixel_coords, img_id


def frame_order(path):
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

    # arrange frames by position in scan
    slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
    return ordered_slices


def get_contours_dict(contour_file, path):
    """
    Returns a dictionary as k: img fname, v: [corresponding img_arr, corresponding contour_arr]
    Inputs:
        contour_file: .dcm contour file name
        path: path which has contour and image files
    Returns:
        contour_dict: dictionary with 2d np.arrays
    """
    f = dicom.read_file(os.path.join(path, contour_file))

    # get roi names
    roi_list = [x.ROIName for x in f.StructureSetROISequence]

    # get contours by image id for each roi
    contours_dict = {}
    for index, roi in enumerate(f.ROIContourSequence):
        # get contour datasets in a list
        contours = [contour for contour in roi.ContourSequence]
        # get list of (contour_arr, im_id)
        img_contour_arrays = [coord2pixels(cdata, path) for cdata in contours]

        # dict {image id:list of contours} for each roi
        roi_dict = defaultdict(list)
        for cntr_set, im_id in img_contour_arrays:
            roi_dict[im_id].append(cntr_set)

        contours_dict[roi_list[index]] = roi_dict

    return contours_dict


def get_data(path):
    """
    Generate image array and contour array
    Inputs:
        path (str): path of the the directory that has DICOM files in it
    """

    # get contour file
    contour_path, contour_file = get_contour_file(path)
    # get slice orders
    ordered_frames = frame_order(path)

    # build array of images and get metadata
    images, patient_id, modality, position = [], None, None, None
    for i, (k, v) in enumerate(ordered_frames):
        dcm = dicom.read_file(os.path.join(path, k + '.dcm'))
        if i == 0:
            patient_id = dcm.PatientID
            modality = dcm.Modality
            position = dcm.PatientPosition

        # get image array
        img_arr = parse_dicom_image(dcm)
        images.append(img_arr)
    assert None not in [patient_id, modality, position]

    data_dict = {'patientid': patient_id, 'modality': modality, 'position': position,
                 'contours': {}, 'ordered_uids': [uid for uid, pos in ordered_frames]}

    # get contours for each roi
    roi_contours = get_contours_dict(contour_file, path)

    # assign roi contours to frames
    for roi, cdict in roi_contours.items():
        data_dict['contours'][roi] = []
        for index, (k, v) in enumerate(ordered_frames):
            # load contour for this roi for this frame
            if k in cdict:
                data_dict['contours'][roi].append(cdict[k])
            # no contour for this roi for this frame
            else:
                data_dict['contours'][roi].append([])

    return images, data_dict
