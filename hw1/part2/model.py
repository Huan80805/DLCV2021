import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

# TO FCN: modift dense network to 3 conv+transpose conv
# To remain size: kernel=1
class VGG16_FCN32(nn.Module):
    def __init__(self):
        super(VGG16_FCN32, self).__init__()
        self.vgg16_bn = models.vgg16_bn(pretrained=True)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.conv8 = nn.Conv2d(4096, 7, 1)
        self.up32 =  nn.ConvTranspose2d(7, 7, 64, stride=32, bias=False)
    
    def forward(self, x):
        origin_shape = x.shape
        #print(x.shape) (3,512,512)
        x = self.vgg16_bn.features(x)
        #print(x.shape) (512,16,16)
        x = self.conv6(x)
        #print(x.shape) (4096,16,16)
        x = self.conv7(x)
        #print(x.shape) (4096,16,16)
        x = self.conv8(x)
        #print(x.shape) (7,16,16)
        x = self.up32(x)
        #print(x.shape) (7,544,544), truncate to 512*512
        x = x[:, :, 16: 16 + origin_shape[2], 16: 16 + origin_shape[3]]
        return x
# why choose FCN8s: https://arxiv.org/pdf/1605.06211.pdf
class VGG16_FCN8(nn.Module):
    def __init__(self):
        super(VGG16_FCN8, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.conv8 = nn.Conv2d(4096, 7, 1)
        self.pool3 = nn.Conv2d(256, 7, 1)
        self.pool4 = nn.Conv2d(512, 7, 1)
        self.up2_1 =  nn.ConvTranspose2d(7, 7, 4, stride=2, bias=False)
        self.up2_2 =  nn.ConvTranspose2d(7, 7, 4, stride=2, bias=False)
        self.up8 =  nn.ConvTranspose2d(7, 7, 16, stride=8, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        origin_shape = x.shape
        x_8 = self.vgg16.features[:17](x)
        x_16 = self.vgg16.features[17:24](x_8)
        x_32 = self.vgg16.features[24:](x_16)

        x_32 = self.conv6(x_32)
        x_32 = self.conv7(x_32)
        x_32 = self.conv8(x_32)
        x_32 = self.up2_1(x_32)
        #print(x_32.shape) # (7,34,34) should truncate to (7,32,32)
        x_16 = self.pool4(x_16)
        x_16 = x_32[:, :, 1:-1, 1:-1] + x_16
        x_16 = self.up2_2(x_16)
        #print(x_16.shape) #(7,66,66) should truncate to (7,64,64)
        x_8 = self.pool3(x_8)
        x_8 = x_16[:, :, 1:-1, 1:-1] + x_8
        x = self.up8(x_8)
        #print(x.shape) # (7,520,520) should truncate to (7,512,512)
        x = x[:, :, 4:-4, 4:-4]
        return x

if __name__ == "__main__":
    model = VGG16_FCN8()
    #print(model)
    #print(model.vgg16_bn)
    test_tensor = torch.randn(1,3,512,512)
    test_output = model(test_tensor)
    print(test_output.shape)
    
