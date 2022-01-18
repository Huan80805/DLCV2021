import os
import torch
import random
import wandb
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from byol_pytorch import BYOL
from torchvision import models
from utils import save_checkpoint
from dataset import ImageDataset
from config import PretrainConfig

#---------environment setting---------
config = PretrainConfig()
SEED = config.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def pretrain(args):
    #--------log-------
    wandb.init("hw4-part2", config=config)
    wandb.config.update(args)
    wandb.config.update({'device':device})
    #-------------prepare data------------
    trainset = ImageDataset(args.train_csv, 
                            args.train_data_dir,
                            transform=False)
                            # BYOL already implemented augmentation
    train_dataloader = DataLoader(trainset, 
                                batch_size=config.batch_size,
                                shuffle=True)
    #------------build model-------------
    resnet = models.resnet50(pretrained=False).to(device)
    # reference: optimal parameter in BYOL repo 
    net = BYOL(
        resnet,
        image_size=128,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    ).to(device)

    optim = torch.optim.Adam(net.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    if args.resume:
        ckpt = torch.load(args.pretrained_ckpt_path)
        net.load_state_dict(ckpt['state_dict'])
        optim.load_state_dict(ckpt['optimizer'])
    #---------------train-------------
    best_loss = 1e4
    log_loss = 0.
    save_loss = 0.
    step = 0
    for epoch in range(config.epochs):
        net.train()
        for batch_idx, (images,_) in enumerate(train_dataloader):
            step += 1
            optim.zero_grad()
            images = images.to(device)
            loss = net(images)
            loss.backward()
            optim.step()
            net.update_moving_average()
            # -------Statistic-------
            log_loss += loss.item()
            save_loss += loss.item()
            if (step+1) % args.log_interval == 0 :
                log_loss /= args.log_interval
                print('[Train] Epoch: {} [{}/{} ({:.0f}%)] loss:{:.6f}'.format(
                epoch, batch_idx * config.batch_size, 
                len(trainset),
                100.*batch_idx*config.batch_size/len(trainset),
                log_loss))
                wandb.log({"TRAIN_LOSS": log_loss})
                log_loss = 0
            if (step+1) % args.save_interval == 0:
                save_loss /= args.save_interval
                if save_loss < best_loss:
                    best_loss = save_loss
                    ckpt_path = os.path.join(args.ckpt_dir,"BYOL_step{}.pth".format(step))
                    save_checkpoint(ckpt_path,net,optim)
                    print("saving best model at step{}".format(step))
                save_loss = 0.