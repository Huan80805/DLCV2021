import torch.nn as nn
from torchvision import models

class Resnet(nn.Module):
    def __init__(self, pretrained=None, freeze=False, hidden_dim=4096):
        super(Resnet, self).__init__()
        if pretrained is None:
            resnet = models.resnet50(pretrained=False)
        else:
            resnet = pretrained
        if freeze:
            for name, child in resnet.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        num_classes = 65 # of office home classes
        modules = list(resnet.children())[:-1]
        # print(list(resnet.children()))
        # print("---------------------------")
        # print(modules)
        self.backbone = nn.Sequential(*modules)
        print(self.backbone)
        # expand?
        self.classifier = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), 2048)
        out = self.classifier(x)
        return out


if __name__ == '__main__':
    # resnet = models.resnet50(pretrained=False)
    # from torchsummary import summary
    # summary(resnet, (3, 128, 128))
    # print(resnet)
    # for name, child in resnet.named_children():
    #     print(name)
    resnet = Resnet()