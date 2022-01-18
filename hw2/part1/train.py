from torch.optim import optimizer
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
import argparse
import os
from model import Generator, Discriminator
from dataset import dataset
wandb.init("faster")

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_epoch', type=int, default=5)
parser.add_argument('--lr_D', type=float, default=5e-5)
parser.add_argument('--lr_G', type=float, default=2e-4)
parser.add_argument('--lr_decay_rate', type=float, default=0.5)
parser.add_argument('--lr_decay_epoch', type=int, default=50)
parser.add_argument('--lr_scheduler', type=str, default='step')
parser.add_argument('--dataset_path', type=str, default='../hw2_data/face/train')
parser.add_argument('--smoothed', type=bool, default=False, help='Utilize smoothed labels or not')
parser.add_argument('--ckpt_path', type=str, default='model/')
config = parser.parse_args()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
      
# -----useful tips: https://github.com/soumith/ganhacks--------------
# -----tips1: images to [-1,1]---------------------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])

trainset = dataset(root=config.dataset_path,transform=train_transform)
print(len(trainset))
trainset_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
# -----Initialize weight with zero-centered, 0.02 stadard deviation normal distribution.-----
# -----Ref:https://arxiv.org/pdf/1511.06434.pdf page3----------------------------------------
# -----Ref:https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html------------------


use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)
G = Generator()
D = Discriminator()
G.to(device)
D.to(device)
G.apply(weights_init)
D.apply(weights_init)
criterion = nn.BCELoss() #discriminator
optim_G = optim.Adam(G.parameters(), lr=config.lr_G, betas=(0.5,0.999))
optim_D = optim.Adam(D.parameters(), lr=config.lr_D, betas=(0.5,0.999))
optim_G_scheduler = optim.lr_scheduler.StepLR(optim_G, step_size=config.lr_decay_epoch, gamma=config.lr_decay_rate)
optim_D_scheduler = optim.lr_scheduler.StepLR(optim_D, step_size=config.lr_decay_epoch, gamma=config.lr_decay_rate)

if config.smoothed:
    real_label, fake_label = 0.9,0.1
else:
    real_label, fake_label = 1.0,0.0
def train(G,D, trainset_loader, epochs, log_interval):
    G.train()
    D.train()
    iteration = 0
    D_realloss_total = 0
    D_fakeloss_total = 0
    G_loss_total = 0
    for ep in range(epochs):
        for batch_idx, real_data in enumerate(trainset_loader):
            #-------TRAINING DISCRIMINATOR-----
            real_data = real_data.to(device)
            optim_D.zero_grad()
            label = torch.full((real_data.size(0),),real_label,dtype=torch.float).to(device)
            output = D(real_data).view(-1)
            D_realloss = criterion(output, label)
            #D_realloss.backward()
            z = torch.randn(real_data.size(0),100,1,1).to(device)
            fake_data = G(z)
            label = torch.full((real_data.size(0),),fake_label,dtype=torch.float).to(device)
            #detach: keep generator fixed
            output = D(fake_data.detach()).view(-1)
            D_fakeloss= criterion(output,label)
            #D_fakeloss.backward()
            D_loss = (D_realloss + D_fakeloss)*0.5
            D_loss.backward()
            optim_D.step()
            #----------TRAINING GENERATOR---------
            optim_G.zero_grad()
            output = D(fake_data).view(-1)
            # Not smoothing for generator
            label = torch.full((real_data.size(0),),1.0,dtype=torch.float).to(device)
            G_loss = criterion(output,label)
            G_loss.backward()
            optim_G.step()
            D_realloss_total += D_realloss
            D_fakeloss_total += D_fakeloss
            G_loss_total += G_loss
            if iteration % log_interval == 0 and iteration > 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLOSS: [D_real:{:.6f}, D_fake:{:.6f}, G:{:.6f}]'.format(
                   ep, batch_idx * len(real_data), 
                   len(trainset_loader.dataset),
                   100. * batch_idx / len(trainset_loader),
                   D_realloss_total.item()/(real_data.size(0)*config.log_interval),
                   D_fakeloss_total.item()/(real_data.size(0)*config.log_interval),
                   G_loss_total.item()/(real_data.size(0)*config.log_interval)))
                wandb.log({
                "G_LOSS":G_loss_total.item()/(real_data.size(0)*config.log_interval),
                "D_REALLOSS":D_realloss_total.item()/(real_data.size(0)*config.log_interval),               
                "D_FAKELOSS":D_fakeloss_total.item()/(real_data.size(0)*config.log_interval)               
                })
                D_realloss_total = 0
                D_fakeloss_total = 0
                G_loss_total = 0
            iteration += 1
        #test(model, valset_loader)
        if ep % 5 == 0 :
            save_checkpoint(os.path.join(config.ckpt_path,'G_epoch{}.pth'.format(ep)), G, optim_G)
            save_checkpoint(os.path.join(config.ckpt_path,'D_epoch{}.pth'.format(ep)), D, optim_D)
    # save the final model
    save_checkpoint(os.path.join(config.ckpt_path,'G_epoch{}.pth'.format(ep)), G, optim_G)
    save_checkpoint(os.path.join(config.ckpt_path,'D_epoch{}.pth'.format(ep)), D, optim_D)


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

train(G,D, trainset_loader, epochs=config.epochs, log_interval=config.log_interval)
