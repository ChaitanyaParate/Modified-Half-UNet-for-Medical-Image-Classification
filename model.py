import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),

            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class classification(nn.Module):
    def __init__(self, input_dim):
        super(classification, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 10),
            nn.ReLU(),

            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.linear(x)
    
class H_UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature=64):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = None
        
        
        self.final_conv = nn.Identity()

        for _ in range(4):
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature


    def forward(self, x):

        for down in self.downs:
            x = down(x)
            
            x = self.pool(x)

        x = self.final_conv(x)

        x = torch.flatten(x, start_dim=1)

        if self.classifier is None:
            input_dim = x.shape[1]
            
            self.classifier = classification(input_dim).to(x.device)
        x = self.classifier(x)

        return x
    
def test():
    x = torch.randn((3, 1, 161, 161))
    model = H_UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.view)
    assert preds.shape == (x.shape[0], 1)

if __name__ == "__main__":
    test()