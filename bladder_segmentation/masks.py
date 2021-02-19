import numpy as np
import scipy
import skimage
from skimage import feature, measure, segmentation

from utils import contour2mask


class SobelMask:
    def __init__(self):
        self.name = "Sobel"

    @staticmethod
    def compute_mask(img):
        mask = skimage.filters.sobel(img)
        markers = np.zeros_like(img)
        # set to background marker
        markers[img < 8000] = 1
        # set to bladder marker
        markers[img > np.max(img) * .15] = 2
        # separate bladder from background
        mask = skimage.segmentation.watershed(mask, markers)
        mask[mask == 1] = 0
        mask[mask == 2] = 1
        # remove objects smaller than min size
        mask = skimage.morphology.remove_small_objects(mask.astype(bool), min_size=30)

        canny = CannyMask.compute_mask(img)
        if mask.sum() < 5:
            mask = canny

        return mask


class CannyMask:
    def __init__(self):
        self.name = "Canny"

    @staticmethod
    def compute_mask(img):
        mask = skimage.feature.canny(img, low_threshold=5000, sigma=4.2)
        markers = np.zeros_like(img)
        markers[img < 8000] = 1
        markers[img > np.max(img) * .25] = 2
        mask = skimage.segmentation.watershed(mask, markers)
        mask[mask == 1] = 0
        mask[mask == 2] = 1
        mask = scipy.ndimage.morphology.binary_fill_holes(mask)

        if mask.sum() > 100:
            mask = skimage.morphology.remove_small_objects(mask.astype(bool), 30)
        else:
            mask = skimage.morphology.remove_small_objects(mask.astype(bool), 25)

        return mask


class MarchSquaresMask:
    def __init__(self):
        self.name = "Marching-Squares"

    @staticmethod
    def compute_mask(img, level=21000):
        contours = measure.find_contours(img, level=level)  # TODO: try mask param in find_contours

        if len(contours) == 0:
            mask = np.zeros(img.shape, int)

        else:
            # get the largest contour which should be the bladder
            list_len = [len(i) for i in contours]
            contours = [contours[np.argmax(list_len)]]
            # convert to array of ints
            contours = np.rint(contours).astype(int)
            # convert to mask
            mask = contour2mask(contours, img.shape, xy_ordering=False)  # FIXME: when view is x and y, get just an outline...

        return mask


class EnsembleMeanMask:
    def __init__(self):
        self.name = "Ensemble-Mean"

    @staticmethod
    def compute_mask(img):
        mask = [CannyMask.compute_mask(img),
                SobelMask.compute_mask(img),
                MarchSquaresMask.compute_mask(img)]

        mask = sum([m.astype(int) for m in mask]) / len(mask)
        mask = np.round(mask).astype(int)

        return mask


class EnsembleUnionMask:
    def __init__(self):
        self.name = "Ensemble-Union"

    @staticmethod
    def compute_mask(img):
        mask = [CannyMask.compute_mask(img),
                SobelMask.compute_mask(img),
                MarchSquaresMask.compute_mask(img)]

        mask = sum([m.astype(int) for m in mask])
        mask[mask > 1] = 1

        return mask


class DummyMask:
    def __init__(self):
        self.name = "Dummy"

    @staticmethod
    def compute_mask(img):
        return np.ones(img.shape, int)
