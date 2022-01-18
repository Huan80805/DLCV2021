import torch 
import torch.nn as nn
import numpy as np
def distance_metric(x, y, metric=None):
    # x: query, (n_query*n_way, out_channels)
    # y: proto, (n_way, out_channels)
    # output: distance, (n_way, n_way*n_query)
    if metric == 'cosine':
        w = x.size(0)
        qw = y.size(0)
        x = x.unsqueeze(1).expand(w, qw, -1)
        y = y.unsqueeze(0).expand(w, qw, -1)
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        # CEL toward maximum at true label
        # maximum : similarity=1
        return cos(x,y)
    elif metric == 'euclidian':
        w = x.size(0)
        qw = y.size(0)
        a = x.unsqueeze(1).expand(w, qw, -1)
        b = y.unsqueeze(0).expand(w, qw, -1)
        distance = ((a - b)**2).sum(dim=2)
        # CEL toward maximum at true label
        # maximum : distance=0
        return -distance
    else:
        raise Exception("Distance Metric Error")

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('--------model saved to %s-------' % checkpoint_path)


if __name__ =="__main__":
    n_way, n_query = 5, 15
    out_channels = 1600
    y = torch.rand((n_way, 1600))
    x = torch.rand((n_query*n_way, 1600))
    distance = distance_metric(x, y, metric='cosine')
    print(distance.shape)
    distance = distance_metric(x, y, metric="euclidian")
    print(distance.shape)