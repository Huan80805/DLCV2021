import torch.nn as nn
import torchvision.models as models
# change class: 50

class vgg16_mod(nn.Module):
    def __init__(self):
        super(vgg16_mod, self).__init__()
        #VGG16_bn: add BatchNormalize to VGG16
        self.vgg16 = models.vgg16_bn(pretrained=True)        
        self.vgg16.classifier[6] = nn.Linear(4096,50,bias=True)

    def forward(self, x):
        x = self.vgg16(x)
        return x