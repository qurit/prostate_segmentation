# Dataset

## Summary
The dataset included 59 male patients, of which 56 were diagnosed with prostate cancer. The median patient age was 69 (range 46-82) and lesion-positive patients were reported to have one to three lesions, with 42 patients having one lesion, 13 patients having two lesions, and one patient having three lesions. A GE Discovery 600 or 690 scanner (GE Healthcare, USA) was used to generate $[^{18}F]$ DCFPyL PET/CT images for each subject. The PET and CT axial slice dimensions were $192 \times 192$ and $512 \times 512$ respectively. Scans consisted of either 299 or 335 slices depending on patient height. The voxel size for PET was $3.65 \times 3.65 \times 3.27$ mm and for CT it was $0.98 \times 0.98 \times 3.27$ mm.

Contours of the regions of interest (ROI) were manually generated by an expert physician with the aid of MIM Software and were treated as the ground truth segmentation labels. The urethra, prostate lesions, and bladder were contoured using the PET scan. A gradient-based edge detection tool (called "PET edge") was used to efficiently delineate the bladder from adjacent structures with high accuracy. The CT scan, on the other hand, was used for delineating the seminal vesicles and prostate gland. An interpolation tool was used to interpolate across a series of border axial slices to delineate the prostate.

The image below are maximum intensity projection (MIP) images for a sample patient along each of the anatomical planes. The first row is the PET scan and the second row is CT. The contours of the bladder, prostate, and lesions are shown in cyan, yellow, and magenta respectively. Subfigures were generated via `python -m seg_3d.data.mip`.
![](https://github.com/qurit/prostate_segmentation/blob/a539324d1eff217493e759efcb29e9307e52979c/docs/figures/mip.png)

## Dataset structure and contents
The dataset is stored in **data/image_dataset** and was structured using the script [dicom_code/dataset_refactor.py](https://github.com/qurit/prostate_segmentation/blob/1db894b648816fd343518d2591b981a2a7b7de04/dicom_code/dataset_refactor.py). Each patient has an associated subdirectory with two subdirectories **CT** and **PT** for the CT and PET scans respectively. Each of these subdirectories contain the raw dicom files for the axial slices of the scan ordered by the number in the dicom file name.
```
data/image_dataset/
│
├─ global_dict.json
├─ PSMA-01-018/
│   ├─ CT/
│       ├─ contour_dict.json
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
└─ PSMA-01-852/
    ├─ CT/
        ├─ contour_dict.json
        ├─ 0.dcm
        ├─ 1.dcm
        ...
        └─ 334.dcm
    └─ PT/
        ├─ contour_dict.json
        ├─ 0.dcm
        ├─ 1.dcm
        ...
        └─ 334.dcm
```
The JSON file **global_dict.json** stores all patient metadata and contour data and is structured in a similar way as the dataset directory. Each axial slice has a corresponding contour data entry in the list `global_dict[PATIENT_NAME][MODALITY]["rois"][ROI]` for a particular patient, modality, and region of interest (ROI). An empty list, i.e. `[]`, means there is no contour data for that particular slice, modality, and ROI. The unique identifiers of the dicom files in the original dataset are found at `global_dict[PATIENT_NAME][MODALITY]["ordered_uids"]`.
```json
# data/image_dataset/global_dict.json 
{
    "PSMA-01-018": {
        "CT": {
            "fp": "image_dataset/PSMA-01-018/CT",
            "rois": {
                "Inter": [
                    [],
                    ...
                ],
                ...
            },
            "ordered_uids": [
                ...
            ]
        },
        "PT": {
            "fp": "image_dataset/PSMA-01-018/PT",
            "rois": {
                "Bladder": [
                    [],
                    ...
                ],
                ...
            },
            "ordered_uids": [
                ...
            ]
        }
    },
    ...
}
```
Each PET / CT scan directory also contains a JSON file **contour_dict.json** which contains metadata and contour data for that particular patient's scan. This file only exists for redundancy or in case a global index for all scans provided by **global_dict.json** is not needed.


List of the contour data identifiers in `global_dict[PATIENT_NAME][MODALITY]["rois"]`:
- *Bladder*
- *Inter* (prostate drawn using the interpolation tool which is considered the manual draw or ground truth prostate segmentation)
- *Threshold* (prostate drawn using the threshold tool which performed poorly for some patients but provided an efficient and rough segmentation of the prostate with 0.63 dice score averaged across whole dataset)
- *Tumor* (prostate lesion)
- *Tumor2* (second prostate lesion, only applies to certain patients)
- *Tumor3* (third prostate lesion, only applies to certain patients)
- *TURP urethra*
- *R seminal*
- *L seminal*

### Secondary Dataset
In addition to the dataset described above, there is an additional dataset stored in **data/image_dataset_JGH**. In this dataset, there are 5 patients with a PSMA PET scan (no CT) and bladder annotations. Bladder uptake in these scans is substantially lower compared to the main dataset and is therefore a useful test case for evaluating bladder segmentation performance.

## Patient information
All patient ids:
```
'PSMA-01-844', 'PSMA-01-727', 'PSMA-01-568', 'PSMA-01-486', 'PSMA-01-119', 'PSMA-01-444', 'PSMA-01-733', 'PSMA-01-279', 'PSMA-01-787', 'PSMA-01-822', 'PSMA-01-658', 'PSMA-01-852', 'PSMA-01-519', 'PSMA-01-690', 'PSMA-01-360', 'PSMA-01-419', 'PSMA-01-018', 'PSMA-01-664', 'PSMA-01-105', 'PSMA-01-596', 'PSMA-01-169', 'PSMA-01-211', 'PSMA-01-020', 'PSMA-01-654', 'PSMA-01-518', 'PSMA-01-500', 'PSMA-01-770', 'PSMA-01-535', 'PSMA-01-505', 'PSMA-01-561', 'PSMA-01-311', 'PSMA-01-148', 'PSMA-01-820', 'PSMA-01-634', 'PSMA-01-485', 'PSMA-01-143', 'PSMA-01-133', 'PSMA-01-160', 'PSMA-01-331', 'PSMA-01-126', 'PSMA-01-448', 'PSMA-01-045', 'PSMA-01-187', 'PSMA-01-200', 'PSMA-01-326', 'PSMA-01-097', 'PSMA-01-494', 'PSMA-01-732', 'PSMA-01-669', 'PSMA-01-646', 'PSMA-01-514', 'PSMA-01-676', 'PSMA-01-189', 'PSMA-01-771', 'PSMA-01-835', 'PSMA-01-164', 'PSMA-01-715', 'PSMA-01-110', 'PSMA-01-135'
```

Patients with zero annotated lesions:
```
'PSMA-01-200', 'PSMA-01-020', 'PSMA-01-419'
```

 Patients with two or more annotated lesions:
 ```
 'PSMA-01-715', 'PSMA-01-189', 'PSMA-01-646', 'PSMA-01-126', 'PSMA-01-820', 'PSMA-01-148', 'PSMA-01-311', 'PSMA-01-561', 'PSMA-01-211', 'PSMA-01-169', 'PSMA-01-596', 'PSMA-01-787', 'PSMA-01-119', 'PSMA-01-844'
 ``` 

Patients with large percentage (>5% of lesion volume) of overlap between annotated lesions and annotated bladder:
```
'PSMA-01-164', 'PSMA-01-535', 'PSMA-01-676', 'PSMA-01-505', 'PSMA-01-126'
```

Patients with relatively (compared to rest of dataset) large annotated lesions:
```
'PSMA-01-326', 'PSMA-01-634', 'PSMA-01-820', 'PSMA-01-664', 'PSMA-01-690'
```

Patients with relatively (compared to rest of dataset) large annotated bladder:
```
'PSMA-01-133', 'PSMA-01-634', 'PSMA-01-561', 'PSMA-01-505', 'PSMA-01-535', 'PSMA-01-654', 'PSMA-01-105', 'PSMA-01-360', 'PSMA-01-733', 'PSMA-01-727'
```
