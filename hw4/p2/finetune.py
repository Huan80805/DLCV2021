import os
import torch
import random
import numpy as np
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision import models
from dataset import ImageDataset
from config import FinetuneConfig
from model import Resnet
from utils import save_checkpoint
from byol_pytorch import BYOL

#---------environment setting---------
config = FinetuneConfig()
SEED = config.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def finetune(args):
    #--------log-------
    wandb.init("hw4-part2", config=config)
    wandb.config.update(args)
    wandb.config.update({'device':device})
    #-------------prepare data------------
    trainset = ImageDataset(args.train_csv, 
                            args.train_data_dir,
                            transform=True)
    train_dataloader = DataLoader(trainset, 
                                batch_size=config.batch_size,
                                shuffle=True)
    valset = ImageDataset(args.val_csv,
                        args.val_data_dir,
                        transform=False)
    val_dataloader = DataLoader(valset, 
                            batch_size=config.batch_size,
                            shuffle=False)

    #------------build model-------------
    if args.pretrained_ckpt_path is not None:
        # load SSL pretrained model
        state_dict = torch.load(args.pretrained_ckpt_path, map_location=torch.device(device))
        backbone = models.resnet50(pretrained=False)
        if 'state_dict' in state_dict.keys():
            byol = BYOL(
                backbone,
                image_size=128,
                hidden_layer='avgpool',
                projection_size=256,
                projection_hidden_size=4096,
                moving_average_decay=0.99
            ).to(device)
            byol.load_state_dict(state_dict['state_dict'])
        else:
            backbone.load_state_dict(state_dict)

    else: backbone=None
    net = Resnet(pretrained=backbone, freeze=args.freeze)
    net.to(device)

    print([[param.shape, param.requires_grad] for param in net.parameters()])
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), 
                            lr=config.lr, 
                            betas=(config.beta1, config.beta2))
    #---------------train-------------
    train_loss = 0.
    train_acc = 0.
    step = 0
    best_acc = 0.
    for epoch in range(config.epochs):
        net.train()
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            step += 1
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            output = net(images)
            pred = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            loss.backward()
            optim.step()
            # -------Statistic-------
            train_loss += loss.item()
            train_acc += torch.mean((pred == labels).to(torch.float)).item()
            if (step+1) % args.log_interval == 0 :
                train_loss /= args.log_interval
                train_acc = train_acc*100./args.log_interval
                print('[Train] Epoch: {} [{}/{} ({:.0f}%)] loss:{:.6f} acc:{:.2f}'.format(
                epoch, batch_idx * config.batch_size, 
                len(trainset),
                100.*batch_idx*config.batch_size/len(trainset),
                train_loss,
                train_acc
                ))
                wandb.log({"TRAIN_LOSS": train_loss, "TRAIN_ACC": train_acc})
                train_loss = 0.
                train_acc = 0.
        val_acc = eval(args, net, val_dataloader)
        if val_acc > best_acc:
            ckpt_path = os.path.join(args.ckpt_dir,"finetune_epoch{}.pth".format(epoch))
            save_checkpoint(ckpt_path,net,optim)
            print("saving best model at epoch{}".format(epoch))
            best_acc = val_acc
# -------------Validation--------   
def eval(args, net, val_dataloader):
    val_loss = 0.
    val_acc = 0.
    total = 0
    net.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_dataloader):
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            pred = torch.argmax(output, dim=1)
            val_loss += criterion(output, labels).item()
            val_acc += torch.mean((pred == labels).to(torch.float)).item()
            total += 1
    val_acc = 100.*val_acc/total
    val_loss /= total
    print('[Validation] loss:{:.6f} Acc:{:.6f}%'.format(
            val_loss,
            val_acc))
    wandb.log({"VAL_LOSS": val_loss, 
        "VAL_ACC": val_acc})
    return val_acc