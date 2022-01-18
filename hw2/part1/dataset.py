from torch.utils.data import Dataset
import glob
import os
from PIL import Image
#dataset:celebA  size:64*64, RGB, train:40000 test:2621
class dataset(Dataset): 
    #root: hw2_data/face/train,test   
    def __init__(self, root, transform=None):
        self.images = None
        self.filenames = []
        self.root = root            
        filenames = glob.glob(os.path.join(root, '*.png'))
        for fn in filenames:
            self.filenames.append(fn)                 
        self.len = len(self.filenames)
        self.transform = transform
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        image = Image.open(image_fn)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

if __name__ == "__main__":
    train_dataset = dataset(root="../hw2_data/face/train")
    test_dataset  = dataset(root="../hw2_data/face/test")
    print(len(train_dataset))
    print(len(test_dataset))
