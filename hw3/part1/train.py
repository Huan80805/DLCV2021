import os
import numpy as np
import torch
import torch.nn as nn
import transformers
from model import VIT_B16, VIT_B32
import wandb


def train(trainloader, valloader, config):
    # use_cuda = torch.cuda.is_available()
    # device = torch.device('cuda:1' if use_cuda else 'cpu')
    device = config.device
    print("Training with {}".format(device))
    os.makedirs(config.ckpt_path, exist_ok=True)
    ckpt_path = os.path.join(config.ckpt_path,config.model_name)
    os.makedirs(ckpt_path, exist_ok=True)
    if not config.resume_train:
        if config.model_name == 'VIT_B16':   
            model = VIT_B16(config=config).to(device)
        elif config.model_name == 'VIT_B32':
            model = VIT_B32(config=config).to(device)
    else:
        if config.model_name == 'VIT_B16': 
            model = VIT_B16(config=config).to(device)
        elif config.model_name == 'VIT_B32':
            model = VIT_B32(config=config).to(device)
        state = torch.load(config.resume_ckpt_path, map_location=torch.device(device))
        model.load_state_dict(state['state_dict'])
        print("resume training from:{}".format(config.resume_ckpt_path))
    print(model)
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.3)
    #criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
    iteration = 0
    loss_total = 0
    acc_total = 0
    best_acc = 0
    for ep in range(config.epochs):
        for batch_idx, (img, label) in enumerate(trainloader):
            model.train()  
            optim.zero_grad()
            img, label = img.to(device), label.to(device)           
            batch_size = img.size(0)
            pred_label = model(img)
            loss = criterion(pred_label, label)
            loss_total += loss
            acc_total += torch.mean((torch.argmax(pred_label, dim=1) == label).to(torch.float)).item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
            optim.step()
            #scheduler.step()
            # Statistic
            if iteration % config.log_interval == 0 and iteration > 0:
                print('[Train] Epoch: {} [{}/{} ({:.0f}%)] loss:{:.6f} Acc:{:.6f}%'.format(
                ep, batch_idx * batch_size, 
                len(trainloader.dataset),
                100. * batch_idx / len(trainloader),
                loss_total/config.log_interval,
                acc_total*100./config.log_interval))
                wandb.log({"TRAIN_LOSS": loss_total/config.log_interval, 
                "TRAIN_ACC": acc_total*100./config.log_interval})
                loss_total = 0
                acc_total = 0
            iteration += 1

        # Evaluation On Validation Set every epoch
        acc = eval(model, valloader, config)
        if ep % config.save_epoch == 0 and ep>0 and acc>best_acc:
            best_acc = acc
            save_checkpoint(os.path.join(ckpt_path,"{}.pth".format(ep)),model,optim)
    save_checkpoint(os.path.join(ckpt_path,"{}.pth".format(ep)),model,optim)


def eval(model, valloader, config):
    # use_cuda = torch.cuda.is_available()
    # device = torch.device('cuda:1' if use_cuda else 'cpu')
    device = config.device
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss = 0.
    acc = 0.        
    batch = 0
    with torch.no_grad():
        for img, label in valloader:
            batch += 1
            img, label = img.to(device), label.to(device)
            pred_label = model(img)
            loss += criterion(pred_label, label).item()
            acc += torch.mean((torch.argmax(pred_label, dim=1) == label).to(torch.float)).item()
    loss /= batch
    acc = 100.*acc/batch
    #acc = 100.*torch.mean((torch.argmax(pred_label, dim=1) == label).to(torch.float)).item()
    print('[Validation] loss:{:.6f} Acc:{:.4f}%'.format(loss,acc))
    wandb.log({"VAL_LOSS": loss,"VAL_ACC": acc})
    return acc

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('--------model saved to %s-------' % checkpoint_path)
