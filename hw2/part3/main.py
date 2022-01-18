import os
import argparse
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataset import dataset
from train import train
import wandb


        
    
        

parser = argparse.ArgumentParser()
#---------SETTING----------
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_epoch', type=int, default=5)
parser.add_argument("--src_only", default=False, action="store_true")
parser.add_argument("--transform_aug", default=False, action="store_true")
#---------HyperParameters----------
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument('--beta1', type=float, default=0.9, help='Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam optimizer')
#---------Filepath----------
parser.add_argument('--dataset_path', type=str, default='../hw2_data/digits')
parser.add_argument('--ckpt_path', type=str, default='ckpt/')
parser.add_argument('--model_name', default='mnistm', type=str)
parser.add_argument('--source', type=str,default='mnistm', choices=['mnistm', 'svhn', 'usps'])
parser.add_argument('--target', type=str,default='none',choices=['none','mnistm', 'svhn', 'usps'])
config = parser.parse_args()
print(config)

wandb.init(project="part3")
#wandb.init(project="part3",config=config)

if config.transform_aug:
    transform = transforms.Compose([
        transforms.ColorJitter(
            brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ColorJitter(
            brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])
src_trainset = dataset(os.path.join(config.dataset_path, config.source),type='train',transform = transform)
src__trainset, src_valset = train_test_split(src_trainset, test_size=0.25, shuffle=True)
src_trainloader = DataLoader(src_trainset, batch_size=config.batch_size, shuffle=True, num_workers=1)
src_valloader = DataLoader(src_valset, batch_size=config.batch_size, shuffle=True, num_workers=1)
print("src_trainset size:{}, src_valset size:{}".format(len(src__trainset),len(src_valset)))
if config.src_only or config.target=='none':
    tgt_trainloader = None
else:
    tgt_trainset = dataset(root=os.path.join(config.dataset_path, config.target),type='train',transform=transform)
    print("tgt_trainset size:{}".format(len(tgt_trainset)))
    tgt_trainloader = DataLoader(tgt_trainset, batch_size=config.batch_size, shuffle=True, num_workers=1)

train(src_trainloader,src_valloader,tgt_trainloader,config)