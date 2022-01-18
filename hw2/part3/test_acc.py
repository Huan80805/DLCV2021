#test acc on model
import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
from dataset import dataset
from model import DANN
# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',     type=str, default='../hw2_data/digits/mnistm')
parser.add_argument('--output_path', type=str, default='usps.csv')
parser.add_argument('--ckpt_path',   type=str, default='ckpt/mnistm_usps.pth')
parser.add_argument('--target', type=str,default='usps',choices=['mnistm', 'svhn', 'usps'])
config = parser.parse_args()
print(config)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])
#load test_dataset
testset = dataset(root=config.img_dir,type="test",transform=transform)
print("testset size:{}".format(len(testset)))
# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)
model = DANN()
model.to(device)
# Load model
state = torch.load(config.ckpt_path, map_location=torch.device(device))
model.load_state_dict(state['state_dict'])
model.eval()


#csv format (same as test.csv) 
corr = 0
corr_class = [0]*10
corr_class_total = [0]*10
with torch.no_grad():
    for data,label in testset:
        data = torch.unsqueeze(data,0)
        data = data.to(device)
        pred_class, pred_domain = model(data)
        pred_class = pred_class.max(1, keepdim=True)[1] # get the index of the max log-probability
        #cal acc
        corr_class_total[label] += 1
        if pred_class == label:
            corr += 1
            corr_class[pred_class] += 1
corr /= len(testset)
print("Testset Accuracy:{}%".format(100.*corr))
for i in range(10):
    print("DIGITS {} Accuracy:{}%".format(i, 100.*corr_class[i]/corr_class_total[i]))
