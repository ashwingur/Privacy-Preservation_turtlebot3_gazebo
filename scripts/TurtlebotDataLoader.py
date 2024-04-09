import numpy as np
from torch.utils.data import Dataset, Sampler
import pandas as pd
from PIL import Image
import os

class TurtlebotImages(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.csv_data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        csv_row = self.csv_data.iloc[idx]       
        img_name = os.path.join(self.image_dir, csv_row['image'])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image) 


        label = TurtlebotImages.angular_velocity_to_label(csv_row['angular_velocity'])
        return image, label
    
    @staticmethod
    def angular_velocity_to_label(w) -> int:
        if float(w) < 0:
            # Right
            return 0
        elif float(w) == 0:
            # Straight
            return 1
        else:
            # Left
            return 2

class CustomSampler(Sampler):
    def __init__(self, data_source, subset_size):
        self.data_source = data_source
        self.subset_size = subset_size
        self.indices = list(range(len(self.data_source)))
    
    def __iter__(self):
        return iter(np.random.choice(self.indices, size=self.subset_size, replace=False))
    
    def __len__(self):
        return self.subset_size