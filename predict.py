import torch
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms as transforms
from model import SARToOpticalGenerator

def run_prediction(input_image_path, output_save_path, model_checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SARToOpticalGenerator().to(device)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    raw_image = Image.open(input_image_path).convert("L")
    input_tensor = transform(raw_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated_tensor = model(input_tensor)
        generated_tensor = generated_tensor * 0.5 + 0.5
        vutils.save_image(generated_tensor, output_save_path)
        print(f"Execution complete. Output target saved to path destination: {output_save_path}")

if __name__ == "__main__":
    pass
