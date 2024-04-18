import time
import numpy as np
from torch.utils.data import Dataset, Sampler
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms

class TurtlebotDataLoader(Dataset):
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


        label = TurtlebotDataLoader.angular_velocity_to_label(csv_row['angular_velocity'])
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
        
    @staticmethod
    def label_to_direction_string(label) -> str:
        if label == 0:
            return "right"
        elif label == 1:
            return "straight"
        elif label == 2:
            return "left"
        else:
            return "INVALID LABEL"
    

class CustomSampler(Sampler):
    def __init__(self, data_source, subset_size):
        self.data_source = data_source
        self.subset_size = subset_size
        self.indices = list(range(len(self.data_source)))
    
    def __iter__(self):
        return iter(np.random.choice(self.indices, size=self.subset_size, replace=False))
    
    def __len__(self):
        return self.subset_size
    

if __name__ == '__main__':
    start_time = time.time()
    '''
    Find out the class distribution of the training data
    '''

    print("Analysing training data overview...")
    # Define transformations
    data_transform = transforms.Compose([
        transforms.Resize((384, 216)),
        transforms.ToTensor(),
    ])

    # Load dataset
    turtlebot_dataset = TurtlebotDataLoader(csv_file='training.csv',
                                image_dir='training',
                                transform=data_transform)
    
    turtlebot_dataloader = DataLoader(turtlebot_dataset, batch_size=1, shuffle=True)

    all_labels = []
    for S in turtlebot_dataloader:
        im, label = S
            
        all_labels += label.tolist()

    print(f'Image input shape: {im.shape}')
    print('Output classes and their counts (0 = right, 1 = straight, 2 = left):')
    print(np.unique(all_labels, return_counts = True))
    print(f'Time taken: {(time.time()-start_time):.2f}s')