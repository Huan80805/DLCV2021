import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import os
from model import Generator
from torch.utils.data import Dataset
from torchvision.transforms import transforms
#Usage: generate 100image for each digit
#Save filename: digit_num.png ex. 0_001.png
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='output/')
parser.add_argument('--ckpt_path',   type=str, default='wgan_g.pth')
config = parser.parse_args()
print(config)
out_transform = transforms.Compose([
    transforms.Resize(28),
])

class testset(Dataset):    
    def __init__(self, z,c):
        self.z = z          
        self.c = c                      
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        return self.z[index], self.c[index]
    def __len__(self):
        """ Total number of samples in the dataset"""
        return len(self.z)
use_cuda = torch.cuda.is_available()
seed = 123
torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)
G = Generator()
G.to(device)
model = torch.load(config.ckpt_path)
G.load_state_dict(model["state_dict"])
G.eval()
batch_size = 32
z_set = torch.randn(1000,100)
c_set = torch.arange(0,10,1)
#c_set = c_set.repeat(100).reshape(100,10).T.reshape(1000)
c_set = c_set.repeat(100)
#z_set = testset(z_set,c_set)
#z_loader = DataLoader(z_set, batch_size=batch_size,shuffle=False)

image = G(z_set.to(device),c_set.to(device))
#for batch_idx, (z,c) in enumerate(z_loader):
#    generated = G(z.to(device),c.to(device))
#    if batch_idx == 0:
#        image = generated.clone()
#    else:
#        image = torch.cat((image,generated),dim=0)

for i in range(100):
    for digit in range(10):
        fn = str(digit) + "_" + (3-len(str(i+1)))*"0" + str(i+1)
        save_image(((image[digit+i*10]+1)/2).clamp(0,1),os.path.join(config.output_path,"{}.png".format(fn)))
save_image(((image+1)/2).clamp(0,1),"total.png",nrow=10)

    




