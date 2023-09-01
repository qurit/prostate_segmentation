import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from seg_3d.data.dataset import ImageToImage3D

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)

def make_pretty_mip(img: np.ndarray, mask: np.ndarray, plane='cor', modality='PET', cmap='Greys_r'):
    assert modality in ['CT', 'PET']
    f, ax = plt.subplots(figsize=(10, 10))

    # ensures we are not modifying the original objects
    img = img.copy()
    mask = mask.copy()

    if plane == 'cor':  # Coronal
        img = np.rot90(np.swapaxes(img, -3, -2), k=2, axes=(-2, -1))
        mask = np.rot90(np.swapaxes(mask, -3, -2), k=2, axes=(-2, -1))

    elif plane == 'sag':  # Sagittal
        img = np.rot90(np.swapaxes(img, -3, -1), k=1, axes=(-2, -1))
        mask = np.rot90(np.swapaxes(mask, -3, -1), k=1, axes=(-2, -1))

    # hack to better visualize image
    if 'PET' in modality:
        q = 0.9990
        img[img > np.quantile(img, q=q)] = np.log(img[img > np.quantile(img, q=q)]) + np.quantile(img, q=q)

    mip = np.amax(img, axis=0)
    mip_im = ax.imshow(mip, cmap=cmap)

    # handle masks
    mips_mask = [np.amax(m, axis=0) for m in mask[1:]]

    for item, color in zip(mips_mask, ['c', 'm', 'y', 'b', 'g', 'r']):
        if not item.sum():
            continue
        contours, _ = cv2.findContours(item.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        xs = [v[0][0] for v in contours[0]]
        ys = [(v[0][1]) for v in contours[0]]
        ax.plot(xs, ys, color=color, linewidth=1)

    if mip.shape[1] > 500:
        ax.set_xticks([*np.arange(0, mip.shape[1], 100)[:-1], mip.shape[1]], [*np.arange(0, mip.shape[1], 100)[:-1], mip.shape[1]])
    else:
        ax.set_xticks([*np.arange(0, mip.shape[1], 50), mip.shape[1]], [*np.arange(0, mip.shape[1], 50), mip.shape[1]])

    if mip.shape[0] > 500:
        ax.set_yticks([*np.arange(0, mip.shape[0], 100)[:-1], mip.shape[0]], [*np.arange(0, mip.shape[0], 100)[:-1], mip.shape[0]])
    else:
        ax.set_yticks([*np.arange(0, mip.shape[0], 50), mip.shape[0]], [*np.arange(0, mip.shape[0], 50), mip.shape[0]])

    if plane == 'cor':  # coronal 
        ax.set_xlabel('Sagittal slice')
        ax.set_ylabel('Axial slice')
    elif plane == 'sag':  # sagittal
        ax.set_xlabel('Coronal slice')
        ax.set_ylabel('Axial slice')
    else:  # transverse/axial
        ax.set_xlabel('Sagittal slice')
        ax.set_ylabel('Coronal slice')

    if plane in ['cor', 'sag']:
        ax.set_aspect(2.6666)

    ax.set_title(modality)
    f.savefig('mip_' + modality + '_' + plane + '.png', dpi=300)


if __name__ == '__main__':
    """
    Run demo via:
    `python -m seg_3d.data.mip`
    """
    print('Loading data...')  # loading data takes time
    dataset = ImageToImage3D(
        dataset_path='data/image_dataset',
        modality_roi_map=[{'PT': ['Bladder', 'Inter', 'Tumor', 'Tumor2', 'Tumor3']},
                          {'CT': ['Bladder', 'Inter', 'Tumor', 'Tumor2', 'Tumor3']}],
        class_labels=['Background', 'Bladder', 'Tumor', 'Inter'],
        num_slices=335,
        slice_shape=(392, 392),
        clamp_ct=None,  # mip of CT is ugly with clamping
    )

    patient = dataset[0]
    im, mask = patient['image'], patient['gt_mask']

    print('patient id', patient['patient'])
    print('image shape', im.shape, 'mask shape', mask.shape)

    im_npy_pet = im.numpy()[0]
    im_npy_ct = im.numpy()[1]
    mask_npy = mask.numpy()

    # PET
    make_pretty_mip(im_npy_pet, mask_npy, plane='cor', modality='PET')
    make_pretty_mip(im_npy_pet, mask_npy, plane='sag', modality='PET')
    make_pretty_mip(im_npy_pet, mask_npy, plane='tra', modality='PET')

    # CT
    make_pretty_mip(im_npy_ct, mask_npy, plane='cor', modality='CT')
    make_pretty_mip(im_npy_ct, mask_npy, plane='sag', modality='CT')
    make_pretty_mip(im_npy_ct, mask_npy, plane='tra', modality='CT')
