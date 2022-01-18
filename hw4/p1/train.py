import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from config import TrainConfig, TestConfig
from model import Convnet, MLP
from utils import distance_metric, save_checkpoint
from dataset import MiniTrainset, TrainSampler, worker_init_fn, MiniTestset, GeneratorSampler

#---------environment setting---------
config = TrainConfig()
tconfig = TestConfig()
SEED = config.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device: ", device)
def train(args):
    #-------exp--------
    config.n_way = args.n_way
    config.n_shot = args.n_shot
    batch_size = config.n_way*(config.n_shot + config.n_query)
    #--------log-------
    wandb.init("hw4-part1", config=config)
    wandb.config.update(args)
    wandb.config.update({'device':device})
    #-------prepare data----
    trainset = MiniTrainset(args.train_csv, args.train_img_dir)
    train_sampler = TrainSampler(label2idx=trainset.label2idx,
                                n_batch=len(trainset)//batch_size,
                                n_way=config.n_way, 
                                n_shot=config.n_shot, 
                                n_query=config.n_query)
    train_dataloader = DataLoader(dataset=trainset, 
                                batch_sampler=train_sampler,
                                pin_memory=False, 
                                num_workers=args.num_workers, 
                                worker_init_fn=worker_init_fn)

    valset = MiniTrainset(args.val_csv, args.val_img_dir)
    val_sampler = TrainSampler(label2idx=valset.label2idx,
                                n_batch=len(valset)//(tconfig.n_way*(tconfig.n_shot+tconfig.n_query)),
                                n_way=tconfig.n_way, 
                                n_shot=tconfig.n_shot, 
                                n_query=tconfig.n_query)
    val_dataloader = DataLoader(dataset=valset, 
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers, 
                            pin_memory=False, 
                            worker_init_fn=worker_init_fn)
    print("valset size:",len(valset))
    print("trainset size:", len(trainset))
    # build model
    feature_extractor = Convnet().to(device)
    criterion = nn.CrossEntropyLoss()
    conv_optim = torch.optim.Adam(feature_extractor.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    if args.distance_metric == 'parametric':
        mlp = MLP().to(device)
        mlp_optim = torch.optim.Adam(mlp.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    else:
        mlp = None
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    train_loss = 0.
    train_acc = 0.
    step = 0
    best_acc = 0.
    for epoch in range(args.epochs):
        # -----------Train----------
        feature_extractor.train()
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            conv_optim.zero_grad()
            images = images.to(device)
            query_labels = torch.arange(config.n_way).repeat(config.n_query).to(torch.long).to(device)
            support_images = images[:config.n_way*config.n_shot]
            query_images   = images[config.n_way*config.n_shot:]
            proto = feature_extractor(support_images)
            proto = proto.reshape(config.n_shot, config.n_way, -1).mean(dim=0)
            query = feature_extractor(query_images)
            if args.distance_metric == 'parametric':
                mlp.train()
                mlp_optim.zero_grad()
                distance = mlp(query, proto)
            else: 
                distance = distance_metric(query, proto, args.distance_metric)
            loss = criterion(distance, query_labels)
            pred = torch.argmax(distance, dim=1)
            acc = torch.mean((pred == query_labels).to(torch.float)).item()
            train_loss += loss
            train_acc += acc
            loss.backward()
            conv_optim.step()
            if args.distance_metric == 'parametric':
                mlp_optim.step()
            # scheduler.step()

            # Statistic
            if (step+1) % args.log_interval == 0 :
                print('[Train] Epoch: {} [{}/{} ({:.0f}%)] loss:{:.6f} Acc:{:.6f}%'.format(
                epoch, batch_idx * batch_size, 
                len(trainset),
                100.*batch_idx*batch_size/len(trainset),
                train_loss/args.log_interval,
                train_acc*100./args.log_interval))
                wandb.log({"TRAIN_LOSS": train_loss/args.log_interval, 
                "TRAIN_ACC": train_acc*100./args.log_interval})
                train_loss = 0.
                train_acc = 0.
            step += 1
        val_acc = eval(args, feature_extractor, mlp, best_acc, val_dataloader)
        if val_acc > best_acc:
            ckpt_path = os.path.join(args.ckpt_dir,"conv_epoch{}.pth".format(epoch))
            save_checkpoint(ckpt_path,feature_extractor,conv_optim)
            if args.distance_metric == 'parametric':
                ckpt_path = os.path.join(args.ckpt_dir,"mlp_epoch{}.pth".format(epoch))
                save_checkpoint(ckpt_path,mlp,mlp_optim)
            print("saving best model at epoch{}".format(epoch))
            best_acc = val_acc
        # if (epoch+1)%5 ==0:
        #     ckpt_path = os.path.join(args.ckpt_dir,"conv_epoch{}.pth".format(epoch))
        #     save_checkpoint(ckpt_path,feature_extractor,conv_optim)
        #     if args.distance_metric == 'parametric':
        #         ckpt_path = os.path.join(args.ckpt_dir,"mlp_epoch{}.pth".format(epoch))
        #         save_checkpoint(ckpt_path,mlp,mlp_optim)
        #     print("saving regular model at epoch{}".format(epoch))


# -------------Validation--------
def eval(args, feature_extractor, mlp, best, val_dataloader):
    feature_extractor.eval()
    val_loss = 0.
    val_acc  = 0.
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_dataloader):
            images = images.to(device)
            #convert string labels to relative labels
            support_images = images[:tconfig.n_way*tconfig.n_shot]
            query_images = images[tconfig.n_way*tconfig.n_shot:]
            images = images.to(device)
            query_labels = torch.arange(tconfig.n_way).repeat(tconfig.n_query).to(torch.long).to(device)
            proto = feature_extractor(support_images)
            proto = proto.reshape(tconfig.n_shot,tconfig.n_way,-1).mean(dim=0)
            query = feature_extractor(query_images)
            if args.distance_metric == 'parametric':
                mlp.eval()
                distance = mlp(query, proto)
            else: 
                distance = distance_metric(query, proto, args.distance_metric)
            loss = criterion(distance, query_labels)
            pred = torch.argmax(distance, dim=1)
            acc = torch.mean((pred == query_labels).to(torch.float)).item()
            val_loss += loss
            val_acc += acc
            total += 1
    print('[Validation] loss:{:.6f} Acc:{:.6f}%'.format(
            val_loss/total,
            100.*val_acc/total))
    wandb.log({"VAL_LOSS": val_loss/total, 
        "VAL_ACC": val_acc*100./total})
    return val_acc
