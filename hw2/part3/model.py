
from torch.autograd import Function
import torch.nn as nn 

class ReverseLayerF(Function):
#ref:https://github.com/fungtion/DANN
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  
        grad_input = grad_output.neg() * ctx.alpha
        
        return grad_input, None

def init_weights(model, normal=True):
    if normal:
        init_func = nn.init.xavier_normal_
        init_zero = nn.init.zeros_
    else:
        init_func = nn.init.xavier_uniform_
        init_zero = nn.init.zeros_

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init_func(m.weight.data)
            if m.bias is not None:
                init_zero(m.bias.data)
        elif isinstance(m, nn.Linear):
            init_func(m.weight.data)
            if m.bias is not None:
                init_zero(m.bias.data)

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, 5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )
        
        self.class_classifier = nn.Sequential(
            nn.Linear( 50*4*4 , 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(50*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def get_latent(self, x):
        latent = self.feature_extractor(x)
        latent = latent.view(-1, 50 * 4 * 4)
        return latent


    def forward(self, x, alpha=1.0):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        
        return class_output, domain_output
