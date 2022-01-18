import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

import numpy as np
#REF:https://www.jianshu.com/p/31a054255fcc
#Initialize a transpose convolution layer
def _bilinear_kernel(in_channels:int, out_channels:int, kernel_size:int) -> torch.Tensor:
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("ConvTranspose") != -1:
        m.weight.data = _bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0])
    elif classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.n_classes = 10
        self.latent_dim = 100
        self.img_size = 28
        self.label_emb = nn.Embedding(self.n_classes, self.latent_dim)
        self.init_size = self.img_size // 4  #7*7
        # 512*7*7
        self.l1 = nn.Linear(self.latent_dim, 512 * self.init_size ** 2)
        # 2*upsample each conv, add thicker by tconv+conv(no downsample)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512,512,2,stride=2),
            nn.Conv2d(512,256,3,stride=1,padding=1),
            nn.BatchNorm2d(256,0.8),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(256,256,2,stride=2),
            nn.Conv2d(256,128,3,stride=1,padding=1),
            nn.BatchNorm2d(128,0.8),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64,0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,3,3, stride=1, padding=1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z, c):
        out = torch.mul(self.label_emb(c), z)
        out = self.l1(out)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_classes = 10
        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2,inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block[0] = nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        self.adv_layer = nn.Sequential(nn.Linear(512*1*1, 1))
        self.aux_layer = nn.Sequential(nn.Linear(512*1*1, self.n_classes))

        self.apply(weights_init)

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        TF = self.adv_layer(out)
        classes = self.aux_layer(out)

        return classes, TF


if __name__ == "__main__":
  G = Generator()
  z = torch.rand((5, 100))
  c = torch.arange(0, 5).type(torch.LongTensor)
  y = G(z, c)
  print(y.shape)
  D = Discriminator()
  classes, TF = D(y)
  print(TF.shape, classes.shape)
  
  
