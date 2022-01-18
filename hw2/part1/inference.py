import random
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Normalize
from torchvision.utils import save_image
import torch
from model import Generator
from torch.utils.data import Dataset
from torchvision.transforms import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='output/')
parser.add_argument('--ckpt_path',   type=str, default='dcgan_g.pth')
config = parser.parse_args()
print(config)
mean = [-0.5/0.5,-0.5/0.5,-0.5/0.5]
std = [2,2,2]
class denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
class testset(Dataset):    
    def __init__(self, data):
        self.data = data          
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        return self.data[index].view(100,1,1)
    def __len__(self):
        """ Total number of samples in the dataset"""
        return len(self.data)
use_cuda = torch.cuda.is_available()
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)
G = Generator()
G.to(device)
model = torch.load(config.ckpt_path)
G.load_state_dict(model["state_dict"])
G.eval()
batch_size = 32
z_set = torch.randn(1000,100)
z_set = testset(z_set)
z_loader = DataLoader(z_set, batch_size=batch_size,shuffle=False)

for batch_idx, z in enumerate(z_loader):
    generated = G(z.to(device))
    if batch_idx == 0:
        image = generated.clone()
        #save_image(((image+1)/2).clamp(0,1), "test32.png",nrow=8)
    else:
        image = torch.cat((image,generated),dim=0)
#denorm = denormalize(mean,std)
#image = denorm(image)
for i in range(1000):
    name = (4-len(str(i+1)))*"0" + str(i+1)
    save_image(((image[i]+1)/2).clamp(0,1),os.path.join(config.output_path,"{}.png".format(name)))
    




def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
