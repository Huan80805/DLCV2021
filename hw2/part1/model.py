import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

# Implementation: DC-GAN
# REF : https://arxiv.org/pdf/1511.06434.pdf
# DC-GAN guidelines:
# 1. Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
#    convolutions (generator).
# 2. Use batchnorm in both the generator and the discriminator.
# 3. Remove fully connected hidden layers for deeper architectures.
# 4. Use ReLU activation in generator for all layers except for the output, which uses Tanh.
# 5. Use LeakyReLU activation in the discriminator for all layers.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        latent_dim = 100  #z
        # 100*1 project and reshape to 4*4*1024
        self.proj = nn.ConvTranspose2d(latent_dim,512,4,1,0,bias=False)
        # ReLU + Batchnorm + Strided for all factional-conv layer
        # upsample*2 at each transpose
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512,256,4,2,1,bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1,bias=False)
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1,bias=False)
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,3,4,2,1,bias=False)
        )
        #use tanh at output layer
        self.out = nn.Tanh()
    def forward(self, z):
        x = self.proj(z)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    # Strided + Batchnorm + LeakyReLU each conv layer
    # input:= image B*3*64*64
        # 2*downsample each conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)
        )
        # shape: 512*(64/2^4)**2
        self.classifier = nn.Sequential(
            nn.Conv2d(512,1,4,1,0,bias=False),
            nn.Sigmoid()
	)

    def forward(self, img):
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)

        return x
if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()
    test = torch.randn(1,100)
    x = generator(test)
    print(x.shape)
    x = discriminator(x)
    print(x.shape)
