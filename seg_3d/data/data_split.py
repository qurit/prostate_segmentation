
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

from seg_3d.data.dataset import ImageToImage3D

# load dataset
dataset = ImageToImage3D(
    'data/image_dataset',
    [{'PT': ['Bladder', 'Tumor', 'Tumor2', 'Tumor3']}],
    ['Background', 'Bladder', 'Tumor', 'Tumor2', 'Tumor3'],
    100, None, (128, 128)
)
print(dataset.patient_keys, '\n', len(dataset.patient_keys))

################################################################################
# generate list of bladder and tumor volume sizes
b_vol = []
t_vol = []
for i in range(59):
    p = dataset[i]
    print('\n', p['patient'])
    im, mask = p['image'], p['gt_mask']
    bkg_neg = np.prod(mask.shape[1:]) - mask[0].sum()
    bl = mask[1].sum().item()
    b_vol.append(bl)
    print(bl)
    tu = []
    for i in [2,3,4]:
        v = mask[i].sum().item()
        if v > 0:
            tu.append(v)
    print(tu)
    t_vol.append(tu)

print(b_vol)
print(t_vol)

print(np.sum(b_vol), np.sum([y for x in t_vol for y in x]))

# make histogram
plt.hist(b_vol, bins=100)
plt.title('Bladder volume hist')
plt.figure()
plt.hist(t_vol2, bins=100)
plt.title('Tumor volume hist')

plt.show()

# pickle.dump({'patients': dataset.patient_keys,
#              'bladder_vol': b_vol,
#              'tumor_vol': t_vol},
#             open('patient_roi_volumes.pk', 'wb'))


###############################################################################
# 3 folds, 40-9-10 train-val-test
# every patient is either in a val or test set except for 2 patients
# report results on the 30 test patients
# JGH patients in test set for bladder experiments

idx = np.arange(59, dtype=int)
np.random.shuffle(idx)
folds = {}
patient_keys = np.asarray(dataset.patient_keys)

for i, f in enumerate([idx[0:19], idx[19:38], idx[38:57]]):
    train = [int(item) for item in list(set(idx) - set(f))]
    val = [int(item) for item in list(f[:9])]
    test = [int(item) for item in list(f[9:])] 
    folds[i] = {'train': {'idx': train, 'keys': patient_keys[train].tolist()},
                'val': {'idx': val, 'keys': patient_keys[val].tolist()},
                'test': {'idx': test, 'keys': patient_keys[test].tolist()}}

# print(json.dumps(folds, indent=4))
json.dump(folds, open('data_split.json', 'w'), indent=4)

###############################################################################
# lists of patients of particular interest
significant_bladder_tumor_overlap = ['PSMA-01-164', 'PSMA-01-535', 'PSMA-01-676', 'PSMA-01-505', 'PSMA-01-126']
two_or_more_tumors = ['PSMA-01-715', 'PSMA-01-189', 'PSMA-01-646', 'PSMA-01-126', 'PSMA-01-820', 'PSMA-01-148',
                      'PSMA-01-311', 'PSMA-01-561', 'PSMA-01-211', 'PSMA-01-169', 'PSMA-01-596', 'PSMA-01-787',
                      'PSMA-01-119', 'PSMA-01-844']
zero_tumors = ['PSMA-01-200', 'PSMA-01-020', 'PSMA-01-419']
tumor_volume_greater_than_1000 = ['PSMA-01-326', 'PSMA-01-634', 'PSMA-01-820', 'PSMA-01-664', 'PSMA-01-690']
bladder_volume_greater_than_5000 = ['PSMA-01-133', 'PSMA-01-634', 'PSMA-01-561', 'PSMA-01-505', 'PSMA-01-535',
                                    'PSMA-01-654', 'PSMA-01-105', 'PSMA-01-360', 'PSMA-01-733', 'PSMA-01-727']
