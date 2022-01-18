import os
import numpy as np
import torch
from torch.serialization import save
from torch.utils.data import dataloader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from model import DANN
import wandb


def train(src_trainloader, src_valloader, tgt_trainloader, config):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Training with {}".format(device))
    os.makedirs(config.ckpt_path, exist_ok=True)
    ckpt_path = os.path.join(config.ckpt_path,'{}_{}'.format(config.source,config.target))
    os.makedirs(ckpt_path, exist_ok=True)    
    model = DANN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr,  betas=[config.beta1, config.beta2])
    criterion = nn.NLLLoss()

    iteration = 0
    src_d_loss_total = 0
    tgt_d_loss_total = 0
    c_loss_total = 0
    for ep in range(config.epochs):
        if not (config.src_only or config.target == 'none'):
            tgt_dataiter = iter(tgt_trainloader)
        # data, label = next(src_trainiter)
        for batch_idx, (data, class_label) in enumerate(src_trainloader):
            model.train()  
            optim.zero_grad()
            #------Train On Source------
            data, class_label = data.to(device), class_label.to(device)           
            batch_size = data.size(0)
            p = float(iteration) / (config.epochs*len(src_trainloader.dataset)/config.batch_size)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            domain_label = torch.zeros((batch_size,), dtype=torch.long, device=device)
            pred_class, pred_domain = model(data, alpha)
            c_loss = criterion(pred_class, class_label)
            src_d_loss = criterion(pred_domain, domain_label)
            loss = c_loss + src_d_loss
            #------Train On Target Domain------
            if (config.src_only or config.target == 'none'):
                tgt_d_loss = torch.zeros(1)
            else:
                # note: not using class label on target set]
                # fix error: not divided batch_size
                try:
                    data, _ = next(tgt_dataiter)
                except StopIteration:
                    tgt_dataiter = iter(tgt_trainloader)
                    data, _ = next(tgt_dataiter)    
                data = data.to(device)
                tgt_batch_size = data.size(0)
                domain_label = torch.ones((tgt_batch_size,), dtype=torch.long, device=device)
                _, pred_domain = model(data, alpha)
                tgt_d_loss = criterion(pred_domain, domain_label)
                loss += tgt_d_loss
            loss.backward()
            optim.step()
            # Statistic
            src_d_loss_total += src_d_loss.item()
            tgt_d_loss_total += tgt_d_loss.item()
            c_loss_total += c_loss.item()
            if iteration % config.log_interval == 0 and iteration > 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] D_loss: [SOURCE:{:.6f}, TARGET:{:.6f}] C_loss: {:.6f}'.format(
                ep, batch_idx * batch_size, 
                len(src_trainloader.dataset),
                100. * batch_idx / len(src_trainloader),
                src_d_loss_total/config.log_interval,
                tgt_d_loss_total/config.log_interval,
                c_loss_total/config.log_interval))
                wandb.log({"C_LOSS": c_loss_total,
                "SRC_D_LOSS": src_d_loss_total,
                "TGT_D_LOSS": tgt_d_loss_total}, )
                src_d_loss_total = 0
                tgt_d_loss_total = 0
                c_loss_total = 0
            
            iteration += 1

        # Evaluation On Validation Set every epoch
        eval(model, src_valloader)
        if ep % config.save_epoch == 0:
            save_checkpoint(os.path.join(ckpt_path,"{}.pth".format(ep)),model,optim )
    save_checkpoint(os.path.join(ckpt_path,"{}.pth".format(ep)),model,optim)


def eval(model, src_valloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    criterion = nn.NLLLoss()
    model.eval()
    c_loss = 0
    d_loss = 0
    acc = 0.
    with torch.no_grad():
        for data, class_label in src_valloader:
            batch_size = data.size(0)
            data, class_label = data.to(device), class_label.to(device)
            domain_label = torch.zeros((batch_size,), dtype=torch.long, device=device)
            pred_class, pred_domain = model(data)
            c_loss += criterion(pred_class, class_label).item()
            d_loss += criterion(pred_domain, domain_label).item()
            acc += torch.mean((torch.argmax(pred_class, dim=1) == class_label).to(torch.float)).item()
    c_loss /= len(src_valloader)
    d_loss /= len(src_valloader)
    acc = 100.*acc/len(src_valloader)
    print('[Validation] C_loss:{:.6f} D_loss:{:.6f} Acc:{:.4f}'.format(
        c_loss,d_loss,acc))
    wandb.log({"VAL_D_LOSS": d_loss,
               "VAL_ACC": acc,
               "VAL_C_LOSS": c_loss})


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
