import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# MiniImage and Office-Home are of same formats
# MiniImage: 38400/9600 RGB
# OfficeHome: 3951/406 RGB
class ImageDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=True):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        if transform:
            self.transform = transforms.Compose([
                # pretrain: augmentation already implemented in BYOL
                # only used when finetuning
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4,0.4,0.4),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # convert raw_label to num_label
        self.raw_label = self.data_df["label"].tolist() #raw_label
        self.label = []
        self.filename = self.data_df["filename"].tolist()
        self.raw2num = dict()
        label_set = sorted(set(self.raw_label))
        for num_label, raw_label in enumerate(label_set):
            self.raw2num[raw_label] = num_label
        for idx, raw_label in enumerate(self.raw_label):
            self.label.append(self.raw2num[raw_label])
        self.num_classes = len(label_set)
    def __getitem__(self, index):
        path = self.filename[index]
        label = self.label[index]
        image = Image.open(os.path.join(self.data_dir, path))
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.label)

class Testset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.filename = self.data_df["filename"].tolist()
    def __getitem__(self, index):
        path = self.filename[index]
        image = Image.open(os.path.join(self.data_dir, path))
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.filename)


if __name__ == '__main__':
    #trainset 
    pretrainset = ImageDataset('../hw4_data/mini/train.csv', '../hw4_data/mini/train')
    finetunetrainset = ImageDataset('../hw4_data/office/train.csv', '../hw4_data/office/train')
    finetunevalset = ImageDataset('../hw4_data/office/val.csv', '../hw4_data/office/val')
    finetunetestset = Testset('../hw4_data/office/val.csv', '../hw4_data/office/val')
    print(len(pretrainset))  #38400
    print(len(finetunetrainset))  #3951
    print(pretrainset.num_classes)        #64
    print(finetunetrainset.num_classes)        #65
    #save image label2raw
    print(finetunetrainset.raw2num)
    print(finetunevalset.raw2num)  #should be the same
    num2raw = dict()
    for raw_label in finetunetrainset.raw2num:
        num2raw[finetunetrainset.raw2num[raw_label]] = raw_label
    print(num2raw)
    # Save
    np.save('num2raw.npy', num2raw) 
    # Load
    num2raw = np.load('num2raw.npy',allow_pickle='TRUE').item()
    print(num2raw[0])