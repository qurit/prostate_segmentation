import torch
import yaml
from seg_3d.data import dataset
from torch.utils.data import DataLoader
import sys
sys.path.append('./unet_code')
from unet import build_model

cfg_path = '/home/youssef/Desktop/prostate-segmentation/seg_3d/output/SNMMI--bare-multi-mod-bkg-prostate-bladder-all-folds/2/eval_0/config.yaml'
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

print(cfg['DATASET']['CLASS_LABELS'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model(cfg, device)
model.eval()

test_dataset = dataset.InferenceDataset(dataset_path='/home/youssef/Desktop/prostate-segmentation/data/image_dataset')

test_dataloader = DataLoader(test_dataset, batch_size=1)

for sample_idx, test_sample in enumerate(test_dataloader):
    preds = model(test_sample['image']).detach().cpu().numpy()
    print(preds.shape)
    exit()