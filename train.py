import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import SARToOpticalGenerator

class SimulationDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sar_tensor = torch.randn(1, 256, 256)
        optical_tensor = torch.randn(3, 256, 256)
        return sar_tensor, optical_tensor

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training deployment cycle on resource targeted hardware: {device}")

    dataset = SimulationDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    generator = SARToOpticalGenerator().to(device)
    
    l1_loss = nn.L1Loss()
    optimizer_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(1, 2):
        for batch_idx, (sar, optical) in enumerate(dataloader):
            sar, optical = sar.to(device), optical.to(device)
            fake_optical = generator(sar)
            loss_g_l1 = l1_loss(fake_optical, optical) * 100
            
            optimizer_g.zero_grad()
            loss_g_l1.backward()
            optimizer_g.step()

            if batch_idx % 5 == 0:
                print(f"Batch Metrics Processed [{batch_idx}/{len(dataloader)}] -> Model Reconstruction L1 Loss: {loss_g_l1.item():.4f}")

    torch.save(generator.state_dict(), "sar_to_optical_generator.pth")
    print("Project binary schema generated successfully. Target pth file exported safely.")

if __name__ == "__main__":
    train()
