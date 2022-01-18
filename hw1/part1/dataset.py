from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import os
import numpy as np
from PIL import Image

class dataset(Dataset): 
    def __init__(self, root, transform_aug = False):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        if transform_aug == True:
            self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ColorJitter(),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # mean and dev derived from imagenet
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        else:
            self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ]) 
            
        # read filenames
        for i in range(50):
            filenames = glob.glob(os.path.join(root, str(i)+'_*.png'))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
        image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len