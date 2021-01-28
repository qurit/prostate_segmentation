from custom_dicontour import *
import matplotlib.pyplot as plt
import tqdm
import json

root1 = '../Data'  # raw dicom PET-CT data
root2 = 'image_dataset/'  # name of new processed dataset
banned_dir = '../Data/1.2.840.113654.2.70.1.248345942932064946017433599830459061029/' \
             '1.2.840.113654.2.70.1.285864328441314159531538468048958165023'

global_dict = {}
for dirs in tqdm.tqdm(os.listdir(root1)):
    for scandir in os.listdir(os.path.join(root1, dirs)):
        path = os.path.join(root1, dirs, scandir)
        if path == banned_dir:
            continue

        # get image and data dict for a particular patient + scan
        images, data_dict = get_data(path)

        # sanity checks
        for roi in data_dict['contours'].values():
            assert len(roi) == np.shape(images)[0]
        assert np.shape(images)[1:] in [(512, 512), (192, 192)]

        patient_id = data_dict['patientid']
        modality = data_dict['modality']

        # make new dict to store patient-scan data
        new_dir = os.path.join(root2, patient_id, modality)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        with open(os.path.join(new_dir, 'contour_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=4)

        for i in range(np.shape(images)[0]):
            impath = os.path.join(new_dir, str(i)+'.jpeg')
            plt.imsave(impath, images[i], cmap='gray')

        if patient_id in global_dict:
            patient_dict = global_dict[patient_id]
        else:
            patient_dict = {}

        # update global dict
        patient_dict[modality] = {}
        patient_dict[modality]['fp'] = new_dir
        patient_dict[modality]['rois'] = data_dict['contours']
        patient_dict[modality]['ordered_uids'] = data_dict['ordered_uids']

        global_dict[patient_id] = patient_dict

# save global dict to disk
with open(os.path.join(root2, 'global_dict.json'), 'w', encoding='utf-8') as f:
    json.dump(global_dict, f, ensure_ascii=False, indent=4)
