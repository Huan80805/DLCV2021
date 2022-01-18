import torch 
import torch.nn as nn
import numpy as np
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('--------model saved to %s-------' % checkpoint_path)