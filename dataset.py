import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SatelliteDataset(Dataset):
    def __init__(self, sar_dir, optical_dir):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.filenames = [f for f in os.listdir(sar_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]

        self.sar_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.optical_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        sar_path = os.path.join(self.sar_dir, filename)
        optical_path = os.path.join(self.optical_dir, filename)
        
        sar_img = Image.open(sar_path).convert("L")
        optical_img = Image.open(optical_path).convert("RGB")
        
        return self.sar_transform(sar_img), self.optical_transform(optical_img)
