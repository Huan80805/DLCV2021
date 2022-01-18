import torch
import torch.nn as nn
from pytorch_pretrained_vit import ViT 
class VIT_B16(nn.Module) :
    def __init__(self, config):
        super (VIT_B16, self).__init__()
        self.model = ViT('B_16', pretrained=config.pretrained,
                                num_classes=config.num_classes,
                                image_size=config.img_size,
                                dropout_rate=config.dropout_rate,
                                attention_dropout_rate=config.attention_dropout_rate,
                                num_heads=config.num_heads                         
                                )
        
    def forward(self,x):
        x = self.model(x)
        return x

class VIT_B32(nn.Module) :
    def __init__(self, config):
        super (VIT_B32, self).__init__()
        self.model = ViT('B_32', pretrained=config.pretrained, 
                                num_classes=config.num_classes, 
                                attention_dropout_rate=0.1)
        
    def forward(self,x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    test_image = torch.rand((1,3,224,224))
    model = VIT_B32(37)
    model.eval()
    out = model(test_image)
    label = torch.tensor(8,dtype=int).unsqueeze(0)
    print(out)
    criterion = nn.CrossEntropyLoss()
    print(criterion(out, label))