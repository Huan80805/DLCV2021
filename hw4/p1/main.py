import argparse
from train import train
import os
parser = argparse.ArgumentParser(description="Few shot learning")
# Training
parser.add_argument('--resume_ckpt_path', default='best.pth', type=str,
                    help="ckpt to resume training")
parser.add_argument('--log_interval', default=50, type=int,
                    help="validation frequency")
parser.add_argument('--epochs', default=100, type=int,
                    help="small for experiment")                    
# Experiment
parser.add_argument('--n_way', choices=[5,30], type=int, default=30, help="training n_way")
parser.add_argument('--n_shot', choices=[1,5,10], type=int, default=1, help="training n_shot")
parser.add_argument('--distance_metric', choices=['euclidian', 'cosine', 'parametric'], default='parametric',
                    help='distance metric to perform classifying')
# Environment
parser.add_argument('--num_workers', default=4, type=int,
                    help="setting num_workers for dataloader")

# Path
parser.add_argument('--reset', action='store_true',help='reset ckpt dir')
parser.add_argument('--ckpt_dir', default='ckpt/exp', type=str,
                    help="saved_ckpt_dir")
parser.add_argument('--train_csv', default='../hw4_data/mini/train.csv', type=str,
                    help="Training images csv file")
parser.add_argument('--train_img_dir', default='../hw4_data/mini/train', type=str,
                    help="Training images directory")
parser.add_argument('--output_csv', default='../result/p1/train_output.csv', type=str, 
                    help="Output filename")
parser.add_argument('--val_csv', default='../hw4_data/mini/val.csv', type=str,
                    help="Valid images csv file")
parser.add_argument('--val_img_dir', default='../hw4_data/mini/val', type=str,
                    help="Valid images directory")
parser.add_argument('--val_testcase_csv', default='../hw4_data/mini/val_testcase.csv', type=str,
                    help="Valid images directory")

args = parser.parse_args()
if args.reset:
    os.system('rm -rf ' + args.ckpt_dir)
os.makedirs(args.ckpt_dir, exist_ok=True)
train(args)