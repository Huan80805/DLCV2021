import csv
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
from model import Convnet, MLP
from dataset import MiniTrainset, TrainSampler, worker_init_fn
from utils import distance_metric
# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning: Inference")
    parser.add_argument('--mlp_ckpt', default='./ckpt/par_5-1/mlp_epoch112.pth',type=str, 
                        help="mlp path")
    parser.add_argument('--conv_ckpt', default='./ckpt/par_5-1/conv_epoch112.pth',type=str, 
                        help="Convnet path")
    parser.add_argument('--test_csv', default='../hw4_data/mini/val.csv', type=str,
                        help="Test images csv file")
    parser.add_argument('--test_img_dir', default='../hw4_data/mini/val', type=str,
                        help="Test images directory")
    parser.add_argument('--testcase_csv', default='../hw4_data/mini/val_testcase.csv', type=str,
                        help="Test images directory")
    parser.add_argument('--output_csv', default='./output.csv', type=str, 
                        help="Output filename")
    parser.add_argument('--distance_metric', choices=['euclidian','cosine','parametric'],
                        default='parametric')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--n_query', default=15, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    # Init constants:
    args = parse_args()
    n_way, n_shot, n_query = args.n_way, args.n_shot, args.n_query
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)
    ckpt = torch.load(args.conv_ckpt, map_location=torch.device(device))
    feature_extractor = Convnet()
    feature_extractor.to(device)
    feature_extractor.load_state_dict(ckpt['state_dict'])
    feature_extractor.eval()
    if args.distance_metric == 'parametric':
        ckpt = torch.load(args.mlp_ckpt, map_location=torch.device(device))
        mlp = MLP()
        mlp.to(device)
        mlp.load_state_dict(ckpt['state_dict'])
        mlp.eval()
        
    # Prepare dataset
    valset = MiniTrainset(args.test_csv, args.test_img_dir)
    val_sampler = TrainSampler(label2idx=valset.label2idx,
                                n_batch=len(valset)//(args.n_way*(args.n_shot+args.n_query)),
                                n_way=args.n_way, 
                                n_shot=args.n_shot, 
                                n_query=args.n_query)
    val_dataloader = DataLoader(dataset=valset, 
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers, 
                            pin_memory=False, 
                            worker_init_fn=worker_init_fn)
    acc = 0
    count = 0
    for i, (images, labels) in enumerate(val_dataloader):
        count += 1
        images = images.to(device)
        support_images, query_images = images[:n_way*n_shot], images[n_way*n_shot:]
        query_labels = torch.arange(n_way).repeat(n_query).to(torch.long).to(device)
        proto = feature_extractor(support_images)
        proto = proto.reshape(n_shot, n_way, -1).mean(dim=0)
        query = feature_extractor(query_images)
        if args.distance_metric == "parametric":
            distance = mlp(query, proto)
        else:
            distance = distance_metric(query,proto,metric=args.distance_metric)
        pred = torch.argmax(distance, dim=1)
        acc += torch.mean((pred == query_labels).to(torch.float)).item()
    print("validation accuracy:{:.2f}".format(acc*100./count))
        