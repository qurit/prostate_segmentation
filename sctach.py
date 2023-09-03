    import torch
from seg_3d.data import dataset
from torch.utils.data import DataLoader
import sys
sys.path.append('./unet_code')
from unet import build_model

# configure params
cfg = {
    'MODEL': {
        'BACKBONE': {'NAME': 'UNet3D'},
        'META_ARCHITECTURE': 'SemanticSegNet',
        'UNET': {  # UNet params are defined in Abstract3DUNet
            'f_maps': 32,
            'in_channels': 2,
            'out_channels': 3,
        }
    }
}

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model = build_model(cfg, device)
model.eval()

test_dataset = dataset.InferenceDataset(dataset_path='./test', modalities=['PT', 'CT'], crop_size=100,
                                        combined_slice_dims=(192, 192))

test_dataloader = DataLoader(test_dataset, batch_size=1)

for sample_idx, test_sample in enumerate(test_dataloader):
    preds = model(test_sample['image']).detatch()
    print(preds.shape)
    exit()