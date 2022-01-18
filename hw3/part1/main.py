import os
import argparse
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from dataset import dataset
from train import train
import wandb

parser = argparse.ArgumentParser()
#---------SETTING----------
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--log_interval', type=int, default=50)
parser.add_argument('--save_epoch', type=int, default=1)
#---------HyperParameters----------
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument('--beta1', type=float, default=0.9, help='AdamW optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='AdamW optimizer')
parser.add_argument('--weight_decay', type=float, default=0.)
#---------Filepath----------
parser.add_argument('--dataset_path', type=str, default='../hw3_data/p1_data/')
parser.add_argument('--ckpt_path', type=str, default='ckpt/')
#---------Model-------------
parser.add_argument('--img_size', default=224, type=int, choices=[224,384])
parser.add_argument('--num_classes', default=37, type=int)
parser.add_argument('--num_heads', default=12, type=int)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument('--attention_dropout_rate', default=0., type=float)
parser.add_argument('--model_name', default='VIT_B16', type=str, choices=['VIT_B16', 'VIT_B32'])
parser.add_argument('--pretrained', default=False, action='store_true')
parser.add_argument('--resume_train', default=False, action="store_true")
parser.add_argument('--resume_ckpt_path', type=str)
#---------Augmentation------
parser.add_argument('--brightness', default=0., type=float)
parser.add_argument('--saturation', default=0., type=float)
parser.add_argument('--contrast', default=0., type=float)
config = parser.parse_args()
use_cuda = torch.cuda.is_available()
config.device = torch.device('cuda' if use_cuda else 'cpu')
print(config)
wandb.init(project="part1", config=config)
traintransform = transforms.Compose([
    transforms.Resize((config.img_size,config.img_size)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomResizedCrop((224,224),scale=(0.81,1),ratio=(3/4,4/3)),
    transforms.ColorJitter(brightness=config.brightness,contrast=config.contrast,saturation=config.saturation),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
])
valtransform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
])

trainset = dataset(os.path.join(config.dataset_path, "train"),transform = traintransform)
valset = dataset(os.path.join(config.dataset_path, "val"),transform = valtransform)
trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
valloader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
print("trainset size:{}, valset size:{}".format(len(trainset),len(valset)))

train(trainloader,valloader,config)
