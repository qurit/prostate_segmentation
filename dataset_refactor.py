from custom_dicontour import *
import matplotlib.pyplot as plt
import tqdm

root1 = '/home/yous/Desktop/ryt/dicom_dataset/'
root2 = '/home/yous/Desktop/ryt/image_dataset/'
banned_dir = '/home/yous/Desktop/ryt/dicom_dataset/1.2.840.113654.2.70.1.248345942932064946017433599830459061029/' \
             '1.2.840.113654.2.70.1.285864328441314159531538468048958165023'

global_dict = {}
for dirs in tqdm.tqdm(os.listdir(root1)):
    for scandir in os.listdir(os.path.join(root1, dirs)):
        path = os.path.join(root1, dirs, scandir)
        if path == banned_dir:
            continue
        patient_id, modality, images, cont_dict = get_data(path)
        for entry in cont_dict.keys():
            if type(entry) != str:
                assert len(cont_dict[entry]) == np.shape(images)[0]
        assert np.shape(images)[1:] in [(512, 512), (192, 192)]
        new_dir = os.path.join(root2, dirs, scandir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        dict_path = os.path.join(new_dir, 'contour_dict.npy')
        np.save(dict_path, cont_dict)
        for i in range(np.shape(images)[0]):
            impath = os.path.join(new_dir, str(i)+'.jpeg')
            plt.imsave(impath, images[i], cmap='gray')

        if patient_id in global_dict:
            patient_dict = global_dict[patient_id]
        else:
            patient_dict = {}
        patient_dict[modality] = {}
        patient_dict[modality]['fp'] = new_dir
        patient_dict[modality]['rois'] = cont_dict

        global_dict[patient_id] = patient_dict

np.save(os.path.join(root2, 'global_dict.npy'), global_dict)
