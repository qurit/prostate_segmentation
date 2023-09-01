import os
import tqdm
import json
from shutil import copyfile
from dicom_code.custom_dicontour import get_data

RAW_DATA_DIR = 'data/dicom_dataset/data/'   # raw dicom PSMA PET/CT data
NEW_DATA_DIR = 'data/image_dataset2/'       # name of new processed dataset


def dataset_refactor():
    """
    Restructures dataset into a convenient and readible structure:
        
        NEW_DATA_DIR/
        │
        ├─ global_dict.json  // contains metadata and contour data for the scans of each patient
        ├─ PSMA-01-018/
        │   ├─ CT/
        │       ├─ contour_dict.json  // contains metadata and contour data for specific scan and patient
        │       ├─ 0.dcm
        │       ├─ 1.dcm
        │       ...
        │       └─ 334.dcm
        │   └─ PT/
        │       ├─ contour_dict.json
        │       ├─ 0.dcm
        │       ├─ 1.dcm
        │       ...
        │       └─ 334.dcm
        │
        ...

    The contour data of all the scans can easily be accessed by indexing global_dict.json.
    Training pipeline assumes dataset is structured in this way.
    """
    global_dict = {}  # used to store metadata and contour data
    for dirs in tqdm.tqdm(os.listdir(RAW_DATA_DIR)):
        for scandir in os.listdir(os.path.join(RAW_DATA_DIR, dirs)):
            path = os.path.join(RAW_DATA_DIR, dirs, scandir)

            # get image and data dict for a particular patient + scan
            dcm_paths, data_dict = get_data(path)

            # sanity checks
            for roi in data_dict['contours'].values():
                assert len(roi) == len(dcm_paths)

            patient_id = data_dict['patientid']
            modality = data_dict['modality']

            # make new directory to store patient-scan data
            new_dir = os.path.join(NEW_DATA_DIR, patient_id, modality)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            with open(os.path.join(new_dir, 'contour_dict.json'), 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=4)

            for i in range(len(dcm_paths)):
                impath = os.path.join(new_dir, str(i) + '.dcm')
                copyfile(dcm_paths[i], impath)

            patient_dict = global_dict.get(patient_id, {})

            # update global dict
            patient_dict[modality] = {}
            patient_dict[modality]['fp'] = new_dir
            patient_dict[modality]['rois'] = data_dict['contours']
            patient_dict[modality]['ordered_uids'] = data_dict['ordered_uids']

            global_dict[patient_id] = patient_dict

    # save global dict to disk
    with open(os.path.join(NEW_DATA_DIR, 'global_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(global_dict, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # to run via terminal run `python -m dicom_code.dataset_refactor`
    dataset_refactor()
