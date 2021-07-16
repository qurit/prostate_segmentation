import operator

import numpy as np
from scipy import ndimage


def centre_crop(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def dice_score(gt, pred):
    return np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))


def contour2mask(contours, size, xy_ordering=True):
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
