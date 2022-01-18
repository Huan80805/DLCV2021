import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
from model import VGG16_FCN32, VGG16_FCN8
from PIL import Image



if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='../hw1_data/p2_data/validation/')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--model', default='FCN8', type=str)
    parser.add_argument('--ckpt_path')
    config = parser.parse_args()
    print(config)

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
    if config.model == 'FCN8':
        model = VGG16_FCN8().to(device)
    else:
        model = VGG16_FCN32().to(device)
    state = torch.load(config.ckpt_path, map_location=torch.device(device))
    model.load_state_dict(state['state_dict'])

    filenames = glob.glob(os.path.join(config.img_dir, '*.jpg'))
    filenames = sorted(filenames)
    

    model.eval()
    with torch.no_grad():
        for fn in filenames:
            ImageID = os.path.split(fn)[1].split('.')[0]
            output_filename = os.path.join(config.output_dir, '{}.png'.format(ImageID))  
            x = Image.open(fn)
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

