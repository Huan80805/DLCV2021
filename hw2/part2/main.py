import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
import random
import argparse
from model import Generator, Discriminator
from dataset import dataset
from train import train
import numpy as np
import wandb
wandb.init("part2")
parser = argparse.ArgumentParser()
#---------SETTING----------
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_epoch', type=int, default=5)
parser.add_argument('--n_critic', type=int, default=1)
parser.add_argument('--model_name',type=str, default="WGAN", choices=["WGAN","ACGAN"])
parser.add_argument('--class_weight', type=float, default=1.5)
#---------HyperParameters----------
parser.add_argument('--lr_D', type=float, default=4e-4)
parser.add_argument('--lr_G', type=float, default=1e-4)
parser.add_argument('--lr_decay_rate', type=float, default=0.5)
parser.add_argument('--lr_decay_epoch', type=int, default=50)
parser.add_argument('--lr_scheduler', type=str, default='step')
parser.add_argument('--smoothed', type=bool, default=False, help='Utilize smoothed labels or not')
#---------Filepath----------
parser.add_argument('--dataset_path', type=str, default='../hw2_data/digits/mnistm')
parser.add_argument('--ckpt_path', type=str, default='wgan/')
parser.add_argument('--test_folder', type=str, default='wgan/')
config = parser.parse_args()

os.makedirs(config.ckpt_path, exist_ok=True)

print(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device using {}".format(device))
seed = 123
random.seed(seed)
# Numpy
np.random.seed(seed)
# Torch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
G = Generator().to(device)
D = Discriminator().to(device)
print(G)
print(D)


train_transform = transforms.Compose([
transforms.Resize(28),
transforms.ToTensor(),
transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainset = dataset(config.dataset_path,type="train",transform=train_transform)
train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
print(len(trainset))
train(config, train_loader, G, D)






