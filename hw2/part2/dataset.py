from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import csv
#dataset:mnistm  size:28*28, RGB, train:60000 test:10000
class dataset(Dataset): 
    #root: ../hw2_data/digits/mnistm/
    #data: root/train root/test
    def __init__(self, root, type="train", transform=None):
        self.images = None
        self.filenames = []
        self.type = type
        self.root = root
        csv_fn = "train.csv" if (self.type=="train") else "test.csv"
        csv_fn = os.path.join(self.root,csv_fn)
        with open(csv_fn,"r") as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)
            for row in reader:
                row[0] = os.path.join(root,type,row[0])
                row[1] = int(row[1])
                self.filenames.append((row[0],row[1]))                
        self.len = len(self.filenames)
        self.transform = transform
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
        if self.transform is not None:
            image = self.transform(image)

        return image,label

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.filenames)

if __name__ == "__main__":
    train_dataset = dataset(root="../hw2_data/digits/mnistm",type="train")
    test_dataset  = dataset(root="../hw2_data/digits/mnistm",type="test")
    print(train_dataset[0])
    print(len(train_dataset))