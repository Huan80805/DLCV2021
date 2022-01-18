import csv
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
from model import Resnet
from dataset import Testset
# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="SSL learning: Inference")
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--test_csv', default='../hw4_data/office/val.csv', type=str,
                        help="Test images csv file")
    parser.add_argument('--test_data_dir', default='../hw4_data/office/val', type=str,
                        help="Test images directory")
    parser.add_argument('--output_csv', default='./output.csv', type=str, 
                        help="Output filename")
    parser.add_argument('--num2raw', default='./num2raw.npy', type=str, 
                        help="num2raw dictionary path")
    parser.add_argument('--batch_size', default=64, type=int, 
                        help="batch_size")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    num2raw = np.load(args.num2raw,allow_pickle='TRUE').item()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)
    ckpt = torch.load(args.ckpt_path, map_location=torch.device(device))
    net = Resnet()
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
        
    # Prepare dataset
    testset = Testset(args.test_csv, args.test_data_dir)
    test_dataloader = DataLoader(testset, 
                                batch_size=args.batch_size,
                                shuffle=False)
    test_pred = []
    for i, images in enumerate(test_dataloader):
        #label = None in testset
        images = images.to(device)
        output = net(images)
        pred = torch.argmax(output, dim=1).reshape(-1)
        test_pred.extend(pred.tolist())
    header = ['id','filename','label']
    # acc=0
    with open(args.output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for i, row in enumerate(test_pred):
            writer.writerow([i,testset.filename[i],num2raw[row]])
    #         if num2raw[row] in testset.filename[i]:
    #             acc += 1
    # print("accuracy:{:.4f}%".format(100.*acc/len(testset.filename)))
