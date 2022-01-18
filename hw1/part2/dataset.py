import re
import glob
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from PIL import Image

def mask_target(im):
    #7 classes
    #urban(0,1,1) agriculture(1,1,0) rangeland(1,0,1) forest(0,1,0) water(0,0,1) barren(1,1,1) unknown(0,0,0)
    im = transforms.ToTensor()(im)
    im = 4 * im[0] + 2 * im[1] + 1 * im[2]
    #encode0-7: unknown, water, forest, urban, x, rangeland, agriculture, barren   
    target = torch.zeros(im.shape, dtype=torch.long)
    target[im==3] = 0 #urban
    target[im==6] = 1 #agriculture
    target[im==5] = 2 #rangeland
    target[im==2] = 3 #forest
    target[im==1] = 4 #water
    target[im==7] = 5 #barren
    target[im==0] = 6 #unknown
    target[im==4] = 6 #no encoding, set unknown
            
    return target

class dataset(Dataset):
    def __init__(self, root,transform=True, flip=False):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        self.flip = flip

        # sat:_sat.jpg, mask:_mask.png,  must match!
        sat_filenames = glob.glob(os.path.join(root, '*.jpg'))
        sat_filenames.sort()
        for sat in sat_filenames:
            self.filenames.append((sat, sat[:-7]+'mask.png'))
            
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        sat_fn, mask_fn = self.filenames[index]
        sat = Image.open(sat_fn)
        mask = Image.open(mask_fn)
        # sat and mask must flip at the same time
        # (1-0.3)**2 ~ 0.5
        if (self.flip):
            if random.random() > 0.3:
                sat = functional.hflip(sat)
                mask = functional.hflip(mask)

            if random.random() > 0.3:
                sat = functional.vflip(sat)
                mask = functional.vflip(mask)
            
        if self.transform:
            sat = self.transform(sat)

        return sat, mask_target(mask)

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.filenames)
if __name__ == "__main__":
    train_set = dataset("../hw1_data/p2_data/train",flip=True)
    print(len(train_set))
    print(train_set[0][0].shape)

    