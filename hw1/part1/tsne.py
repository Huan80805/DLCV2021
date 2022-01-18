from PIL.Image import Image
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision.transforms import transforms
from matplotlib import markers
from torch.utils.data import DataLoader

from vgg16_mod import vgg16_mod
from dataset import dataset


if __name__ == '__main__':
    # init configs from args

    root = "../hw1_data/p1_data/val_50"
    ckpt_path = "final.pth"
    val_dataset = dataset(root, transform_aug=False)

    model = vgg16_mod()
    state = torch.load(ckpt_path)
    model.load_state_dict(state['state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = model.to(device)

    dataloader = DataLoader(val_dataset)
    x,y = [],[]
    with torch.no_grad():
        for idx,(images, labels) in enumerate(dataloader):
            b = images.size(0)

            images, labels = images.to(device), labels.to(device)

            h = model.vgg16.features(images)
            h = model.vgg16.avgpool(h)
            #print(h.shape)
            
            h = model.vgg16.classifier[:-1](h.view(b,-1))

            latent = h.reshape(b,-1)
            latent = latent.detach().cpu()

            x += latent.numpy().tolist()
            y += labels.cpu().numpy().tolist()

    x = np.array(x)
    y = np.array(y)

    '''t-SNE'''
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    x_tsne = tsne.fit_transform(x)

    # normalize
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    X_norm = (x_tsne - x_min) / (x_max - x_min)
   # show
    NUM_CLASS=50
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/NUM_CLASS) for i in range(NUM_CLASS)]

    fig,ax = plt.subplots(figsize=(15,16))
    for i in range(X_norm.shape[0]):
        if y[i]<NUM_CLASS:
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=colors[y[i]], 
                    fontdict={'weight': 'bold', 'size': 10})
    plt.savefig('tsne.png')
