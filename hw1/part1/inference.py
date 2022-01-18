import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
from vgg16_mod import vgg16_mod



if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',     type=str, default='../hw1_data/p1_data/val_50/')
    parser.add_argument('--output_path', type=str, default='p1_output.csv')
    parser.add_argument('--ckpt_path',   type=str, default='final.pth')
    config = parser.parse_args()
    print(config)

    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    vgg16_mod = vgg16_mod().to(device)

    # Load model
    #state = torch.load(config.ckpt_path, map_location=torch.device(device))
    state = torch.load(config.ckpt_path)
    vgg16_mod.load_state_dict(state['state_dict'])
    vgg16_mod.eval()

    filenames = glob.glob(os.path.join(config.img_dir, '*.png'))
    filenames = sorted(filenames)

    with open(config.output_path, 'w') as f:  
        f.write('image_id,label\n')
        with torch.no_grad():
            for fn in filenames:
                x = Image.open(fn)
                x = valid_transform(x)
                x = torch.unsqueeze(x, 0)
                x = x.to(device)
                output = vgg16_mod(x)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                f.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')
