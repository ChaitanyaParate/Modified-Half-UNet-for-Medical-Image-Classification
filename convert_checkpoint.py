import torch
from model import H_UNET

DEVICE = "cuda"

checkpoint = torch.load("my_checkpoint.pth.tar", map_location=DEVICE)
print("=> Checkpoint keys:", checkpoint.keys())

model = H_UNET(in_channels=1, out_channels=1).to(DEVICE)

_ = model(torch.randn(2, 1, 315, 315).to(DEVICE))

model.load_state_dict(checkpoint["state_dict"])

torch.save(model.state_dict(), "half_unet_brain_tumor.pth")
print("Converted to half_unet_brain_tumor.pth successfully!")
