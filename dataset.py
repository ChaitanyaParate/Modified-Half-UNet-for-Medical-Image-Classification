import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for label, folder in enumerate(["no", "yes"]):
            folder_path = os.path.join(image_dir, folder)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = np.array(Image.open(img_path).convert("L"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, float(label)
