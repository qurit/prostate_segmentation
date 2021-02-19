import os
import numpy as np
import scipy
import pydicom
from scipy.signal import savgol_filter

from utils import centre_crop
from dicom_code.contour_utils import parse_dicom_image


class BladderFrameFinder:

    def __init__(self, crop_size=0.15, relative_frame_ub=0.55, relative_peak_order=0.1):
        self.crop_size = crop_size
        self.relative_frame_ub = relative_frame_ub
        self.relative_peak_order = relative_peak_order

    def find_bladder_frame(self, root, scan):

        bladder = scan['rois']['Bladder']
        search_bound = int(self.relative_frame_ub * len(bladder))

        crop_sums = []

        for frame in range(0, search_bound):

            img_dcm = pydicom.dcmread(os.path.join(root, scan['fp'], str(frame)+'.dcm'))
            orig_img = parse_dicom_image(img_dcm)

            crop_size = (int(self.crop_size * orig_img.shape[0]), int(self.crop_size * orig_img.shape[1]))
            cropped_img = centre_crop(orig_img, crop_size)
            crop_sums.append(cropped_img.sum())

        crop_sums = np.asarray(crop_sums)
        peak_order = int(len(bladder) * self.relative_peak_order)
        pred_frame = scipy.signal.argrelextrema(crop_sums, np.greater, order=peak_order)[0][0]

        return pred_frame, crop_sums
