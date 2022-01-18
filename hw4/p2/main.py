import argparse
import os
from pretrain import pretrain
from finetune import finetune
parser = argparse.ArgumentParser(description="SSL learning")
# Environment setting
parser.add_argument('--num_workers', default=4, type=int,
                    help="setting num_workers for dataloader")
parser.add_argument('--save_interval', default=100, type=int,
                    help='save frequency')         
parser.add_argument('--log_interval', default=20, type=int,
                    help='logger frequency') 
# Path args
parser.add_argument('--reset', action='store_true',
                    help='reset ckpt dir')
parser.add_argument('--ckpt_dir', default='ckpt/exp', type=str,
                    help="saved_ckpt_dir")
#note: change img path when doing finetune/pretraining
parser.add_argument('--train_csv', default='../hw4_data/mini/train.csv', type=str,
                    help="Training images csv file")
parser.add_argument('--train_data_dir', default='../hw4_data/mini/train', type=str,
                    help="Training images directory")
parser.add_argument('--val_csv', default='../hw4_data/office/val.csv', type=str,
                    help="Training images csv file")
parser.add_argument('--val_data_dir', default='../hw4_data/office/val', type=str,
                    help="Training images directory")
parser.add_argument('--output_csv', default='./output.csv', type=str, 
                    help="Output filename")


# Experiment parameters
parser.add_argument('--resume', action='store_true',
                    help='resume pretraining')
parser.add_argument('--pretrain', action='store_true', 
                    help='task: pretrain')
parser.add_argument('--finetune', action='store_true', 
                    help='task: finetune')
parser.add_argument('--pretrained_ckpt_path', type=str,default=None,
                    help='ckpt path being finetuned, if not specified train full model from scratch')
parser.add_argument('--freeze', action='store_true', 
                    help='freeze backbone(otherwise train full), activate only when finetuning')


args = parser.parse_args()
if args.reset:
    os.system('rm -rf ' + args.ckpt_dir)
os.makedirs(args.ckpt_dir, exist_ok=True)
if args.pretrain and args.finetune:
    raise Exception('Choose only one of finetune/ pretrain task')
elif args.pretrain:
    pretrain(args)
elif args.finetune:
    finetune(args)
else:
    raise Exception('Must specify pretrain or finetune')