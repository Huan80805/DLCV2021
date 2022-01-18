
import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
from model import VGG16_FCN32, VGG16_FCN8
from PIL import Image
test_image = ['0010_sat.jpg','0097_sat.jpg','0107_sat.jpg']
ckpt_paths = ['FCN8_0.pth', 'FCN8_20.pth','FCN8_final.pth']
MASK = {
0: (0, 1, 1),
1: (1, 1, 0),
2: (1, 0, 1),
3: (0, 1, 0),
4: (0, 0, 1),
5: (1, 1, 1),
6: (0, 0, 0),
}

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

# Load model
for ckpt_path in ckpt_paths:
    model = VGG16_FCN8()
    state = torch.load(ckpt_path)
    model.load_state_dict(state['state_dict'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        for fn in test_image:        
            ImageID = fn.split('_')[0]
            output_filename = ckpt_path.split('.')[0] + '{}_mask.png'.format(ImageID) 
            filename = os.path.join('../hw1_data/p2_data/validation',fn)  
            x = Image.open(filename)
            x = transform(x)
            image_shape = x.shape
            x = torch.unsqueeze(x, 0)  #(batch, shape)
            x = x.to(device)
            output = model(x)
            pred = output.max(1, keepdim=True)[1].reshape((-1, image_shape[1], image_shape[2]))
            y = torch.zeros((pred.shape[0], 3, pred.shape[1], pred.shape[2]))
            for k, v in MASK.items():
                y[:,0,:,:][pred == k] = v[0]
                y[:,1,:,:][pred == k] = v[1]
                y[:,2,:,:][pred == k] = v[2]

            y = transforms.ToPILImage()(y.squeeze())
            y.save(output_filename)
