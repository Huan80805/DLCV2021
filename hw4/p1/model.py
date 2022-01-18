import torch
import torch.nn as nn


class Convnet(nn.Module):
    #fixed
    #output: n*1600
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels),
        )
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1600, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

    def forward(self, x, y):
        #x: query (n_query*n_way, out_channels)
        #y: proto (n_way, channels)
        #output: distance (n_way*n_query, n_query)
        w = x.size(0)
        qw = y.size(0)
        x = x.unsqueeze(1).expand(w, qw, -1)
        y = y.unsqueeze(0).expand(w, qw, -1)
        return self.fc(x-y).squeeze(-1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from torchsummary import summary
    featuer_extractor = Convnet()
    feature_extractor = featuer_extractor.to(device)
    summary(featuer_extractor, (3, 84, 84))
    test = torch.randn(1, 3, 84, 84).to(device)
    print(featuer_extractor)
    n_way, n_query = 5, 15
    distance = MLP().to(device)
    x = torch.rand((n_way*n_query, 1600)).to(device)
    y = torch.rand((n_way, 1600)).to(device)
    print(distance)