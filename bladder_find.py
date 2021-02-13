import os

import cv2
import numpy as np
import pydicom as dicom
import skimage.filters
import skimage.segmentation
from scipy.signal import argrelextrema, savgol_filter

from utils import centre_crop
from dicom_code.contour_utils import parse_dicom_image


def bladder_finder(scan, root):
    bladder = scan['rois']['Bladder']
    search_range = (int(0.05*len(bladder)), int(.35*len(bladder)))
    diff_btwn_pxl_avgs = []

    for frame in range(*search_range):
        # get image array to get image width, height
        dcm = dicom.read_file(os.path.join(root, scan['fp'], str(frame) + '.dcm'))
        orig_img = parse_dicom_image(dcm)
        _, threshed_img = cv2.threshold(orig_img, 245, 255, cv2.THRESH_BINARY)
        crop_size = (int(0.15 * orig_img.shape[0]), int(0.15 * orig_img.shape[1]))
        cropped_img = centre_crop(threshed_img, crop_size)
        diff_btwn_pxl_avgs.append(np.mean(orig_img) - 2 * np.mean(cropped_img))

    diff_btwn_pxl_avgs = np.asarray(diff_btwn_pxl_avgs)
    smoothed_avg_diffs = savgol_filter(diff_btwn_pxl_avgs, 11, 3)
    pred_frame = np.argmin(smoothed_avg_diffs) + search_range[0]
    return pred_frame


def bladder_bookend(pred_frame, sumset, base_index):
    sumset[sumset != 0] = 1
    sumset = np.asarray(sumset)

    ub = sumset[pred_frame] * 2
    right_half = sumset[pred_frame:]
    r_extrema = np.argwhere(right_half > ub)
    right_frame = r_extrema[0] + pred_frame - 1 + base_index

    l_extrema = argrelextrema(sumset[:pred_frame], np.less, order=1)[0]
    left_frame = l_extrema[-1] - 1 + base_index

    return [left_frame, *right_frame]


def bladder_detect(img):

    img = cv2.GaussianBlur(img, (3, 3), .9)

    _, thresh_pred = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    sobel_pred = skimage.filters.sobel(thresh_pred)
    markers = np.zeros_like(img)
    markers[img < 50] = 1
    markers[img > 120] = 2
    sobel_pred = skimage.segmentation.watershed(sobel_pred, markers)
    sobel_pred[sobel_pred == 1] = 0
    sobel_pred[sobel_pred == 2] = 1
    sobel_pred = skimage.morphology.remove_small_objects(sobel_pred.astype(bool), 30)

    if sobel_pred.sum() == 0.:
        th3 = skimage.filters.sobel(img)
        markers = np.zeros_like(img)
        markers[img < 20] = 1
        markers[img > 80] = 2
        th3 = skimage.segmentation.watershed(th3, markers)
        th3[th3 == 1] = 0
        th3[th3 == 2] = 1
        th3 = skimage.morphology.remove_small_objects(th3.astype(bool), 20)
        sobel_pred = th3

    edges = cv2.Canny(thresh_pred, 50, 200)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    number_of_objects_in_image = len(contours)

    if number_of_objects_in_image >= 7:
        sobel_pred = np.zeros_like(sobel_pred)
    if thresh_pred.sum() / 255 > 1000:
        sobel_pred = np.zeros_like(sobel_pred)

    return sobel_pred
