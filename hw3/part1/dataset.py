from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import os
import PIL.Image as Image
import numpy as np
import cv2
class dataset(Dataset): 
    #root: ../hw3_data/p1_data/train,val
    #37classes
    def __init__(self, root, transform=None):
        self.images = None
        self.root = root
        self.filenames = sorted(glob.glob(os.path.join(root, '*.jpg')))          
        self.len = len(self.filenames)
        self.transform = transform
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        label = os.path.split(image_fn)[-1].split('_')[0]
        image = Image.open(image_fn).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image,int(label)

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.filenames)
if __name__ == "__main__":
    def transform(img):
        transform = transforms.Compose([
            transforms.RandomResizedCrop((224,224),scale=(0.81,0.81),ratio=(3/4,4/3)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=[0.75,0.75],contrast=0.2,saturation=0.2),
            # transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])
        img = transform(img)
        return img
    trainset = dataset("../hw3_data/p1_data/train", transform=transform)
    valset = dataset("../hw3_data/p1_data/val", transform=transform)
    test_img = cv2.imread("../hw3_data/p1_data/train/0_1.jpg")
    print(test_img.shape)
    cv2.imshow("test_img",test_img)
    test_img2 = ((trainset[1][0]+1)/2).numpy()
    print(test_img2.shape)
    cv2.imshow("test_img2",cv2.cvtColor(np.transpose(test_img2,(1,2,0)),cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    # for i in range(len(trainset)):
    #     if trainset[i][0].shape[0] != 3:
    #         print(trainset[i][0].shape)
    
