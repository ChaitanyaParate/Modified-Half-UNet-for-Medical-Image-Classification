import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import H_UNET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = H_UNET(in_channels=1, out_channels=1).to(DEVICE)
_ = model(torch.randn(2, 1, 315, 315).to(DEVICE))
model.load_state_dict(torch.load("half_unet_brain_tumor.pth", map_location=DEVICE))
model.eval()

transform = A.Compose([
    A.Resize(height=315, width=315),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2(),
])

img_path = "test_image.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = torch.sigmoid(model(img))
    result = (pred > 0.5).float().item()

print("Prediction:", "Tumor Detected" if result == 1 else "No Tumor")
