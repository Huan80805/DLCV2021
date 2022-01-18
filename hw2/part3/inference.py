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
parser.add_argument('--img_dir',     type=str, default='../hw2_data/digits/mnistm/test')
parser.add_argument('--output_path', type=str, default='usps.csv')
#parser.add_argument('--ckpt_path',   type=str, default='ckpt/mnistm_usps.pth')
parser.add_argument('--target', type=str,default='usps',choices=['mnistm', 'svhn', 'usps'])
config = parser.parse_args()
print(config)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])
#load test_dataset
# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)
model = DANN()
model.to(device)
# Load model
#state = torch.load(config.ckpt_path, map_location=torch.device(device))
if config.target == 'mnistm':
    state = torch.load('model/svhn_mnistm.pth')
elif config.target == 'usps':
    state = torch.load('model/mnistm_usps.pth')
elif config.target == 'svhn':
    state = torch.load('model/usps_svhn.pth')
else:
    print('Invalid Target')
model.load_state_dict(state['state_dict'])
model.eval()

filenames = glob.glob(os.path.join(config.img_dir, '*.png'))
filenames = sorted(filenames)
#csv format (same as test.csv) 
with open(config.output_path, 'w') as f:  
    f.write('image_name,label\n')
    with torch.no_grad():
        for fn in filenames:
            x = Image.open(fn).convert("RGB")
            x = transform(x)
            x = torch.unsqueeze(x, 0)
            x = x.to(device)
            pred_class, pred_domain = model(x)
            pred_class = pred_class.max(1, keepdim=True)[1] # get the index of the max log-probability
            f.write(os.path.split(fn)[-1] + ',' + str(pred_class.item()) + '\n')
