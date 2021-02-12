import scipy
import operator
import numpy as np
import matplotlib.pyplot as plt


def centre_crop(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def dice_score(gt, pred):
    return np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))


def contour2mask(contours, size):
    mask = np.zeros(size, int)
    poly_set = [np.asarray(poly) for poly in contours]
    for poly in poly_set:
        mask[poly[:, 1], poly[:, 0]] = 1
    mask = scipy.ndimage.morphology.binary_fill_holes(mask)
    return mask


def visualize(img, patient, thresh_pred, sobel_pred, sobel_dice, ground_truth):
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(patient)
    ax1[0].imshow(img)
    ax1[0].set_title('Original ')
    ax1[1].imshow(ground_truth, cmap='Dark2', interpolation='none', alpha=0.7)
    ax1[1].set_title('Ground Truth Mask')
    ax2[0].imshow(thresh_pred)
    ax2[0].imshow(ground_truth, cmap='Dark2', interpolation='none', alpha=0.7)
    ax2[0].set_title('Thresholded '+str(thresh_pred.sum() / 255))
    ax2[1].imshow(sobel_pred, cmap='gray', interpolation='none')
    ax2[1].imshow(ground_truth, cmap='Dark2', interpolation='none', alpha=0.7)
    ax2[1].set_title('Sobel On Original '+str(sobel_dice)+' '+str(sobel_pred.sum()))
    plt.show()
    plt.close('all')
