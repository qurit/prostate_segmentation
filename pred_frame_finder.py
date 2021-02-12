import json
import tqdm
import pydicom
from utils import *
from scipy.signal import savgol_filter
from dataset_building.contour_utils import *


root = '/home/yous/Desktop/ryt/'

with open('/home/yous/Desktop/ryt/image_dataset/global_dict.json') as f:
    data_dict = json.load(f)

correct = 0
for patient in tqdm.tqdm(list(data_dict.keys())):

    scan = data_dict[patient]['PT']
    bladder = scan['rois']['Bladder']
    bladder_frames = [frame for frame, contour in enumerate(bladder) if contour != []]

    bladder_range = (int(0.05*len(bladder)), int(.55*len(bladder)))

    cropped_sums = []

    for frame in range(*bladder_range):

        img_dcm = pydicom.dcmread(os.path.join(root, scan['fp'], str(frame)+'.dcm'))
        orig_img = parse_dicom_image(img_dcm)

        crop_size = (int(0.15 * orig_img.shape[0]), int(0.15 * orig_img.shape[1]))
        cropped_img = centre_crop(orig_img, crop_size)
        cropped_sums.append(cropped_img.sum())

    cropped_sums = np.asarray(cropped_sums) / 10000
    smoothed_avg_diffs = savgol_filter(cropped_sums, 19, 3)
    grad_smooth = np.gradient(cropped_sums)

    pred_frame_raw = np.argmax(cropped_sums)
    pred_frame_smooth = np.argmax(smoothed_avg_diffs)

    if bladder_frames[0] <= pred_frame_raw + bladder_range[0] <= bladder_frames[-1]:
            correct += 1
    else:
        print(patient)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    # fig.suptitle(patient+' '+str((bladder_frames[0], bladder_frames[-1])))
    # ax1.plot(cropped_sums, color='black')
    # ax1.axvline(x=bladder_frames[0] - bladder_range[0], color='black')
    # ax1.axvline(x=bladder_frames[-1] - bladder_range[0], color='black')
    # ax1.axvline(x=pred_frame_raw, color='blue', linestyle='dashed')

    # ax2.plot(smoothed_avg_diffs, color='black')
    # ax2.plot(grad_smooth, color='red')
    # ax2.axvline(x=pred_frame_smooth, color='green', linestyle='dashed')
    # ax2.axvline(x=bladder_frames[0] - bladder_range[0], color='black', linestyle='dashed')
    # ax2.axvline(x=bladder_frames[-1] - bladder_range[0], color='black', linestyle='dashed')
    # for i, infl in enumerate(infls, 1):
    #     if bladder_frames[0] - bladder_range[0] - 5 < infl < bladder_frames[-1] - bladder_range[0] + 5:
    #         ax2.axvline(x=infl, color='b', label=f'Inflection Point {i}')

    # plt.savefig(os.path.join(root, 'bladder_seg1', patient+'.png'))
    # plt.close('all')

print(correct)