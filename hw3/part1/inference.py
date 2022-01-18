import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import glob
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from config import Config
from model import VIT_B16, VIT_B32
class dataset(Dataset): 
    def __init__(self, root, transform=None):
        self.images = None
        self.root = root
        self.filenames = glob.glob(os.path.join(root, '*.jpg'))          
        self.len = len(self.filenames)
        self.transform = transform
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        image = Image.open(image_fn).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_fn
    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.filenames)
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', default='../hw3_data/p1_data/val', type=str)
parser.add_argument('--output_path', default='output.csv', type=str)
parser.add_argument('--img_size', default=384, type=int)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

testtransform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
])
testset = dataset(root=args.img_dir,transform = testtransform)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=1)
configs = [Config('best'),Config('best2'), Config('best6')]
models = []
predictions = []
for config in configs:
    if config.model_name == 'VIT_B16':   
        model = VIT_B16(config=config).to(device)
    elif config.model_name == 'VIT_B32':
        model = VIT_B32(config=config).to(device)
    state = torch.load(config.ckpt_path, map_location=torch.device(device))
    model.load_state_dict(state['state_dict'])
    model.eval()
    models.append(model)
    filenames = []
    with torch.no_grad():
        pred_labels = []
        for image, fn in testloader:
            image = image.to(device)            
            output = model(image)
            pred_label = torch.argmax(output, dim=1).cpu().numpy()
            pred_labels.append(pred_label)
            filenames.extend(fn)
        predictions.append(np.concatenate(pred_labels, axis=0))
predictions = np.array(predictions, dtype=np.uint32).T
out_data = []
for prediction in predictions:
    out_data.append(np.bincount(prediction).argmax())
#write csv
# print(len(filenames))
# print(len(out_data))
# acc = 0
with open(args.output_path, 'w') as f:  
    f.write('filename,label\n')
    for i, fn in enumerate(filenames):
        f.write(os.path.split(fn)[-1] + ',' + str(out_data[i].item()) + '\n')
    #accuracy?
#         if out_data[i].item() == int(os.path.split(fn)[-1].split('_')[0]):
#             acc += 1
# acc = 100.*acc/len(filenames)
# print('ACC:{}%'.format(acc))


