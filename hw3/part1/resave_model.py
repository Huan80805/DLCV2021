# downgrade torch==1.10.0 to 1.2.0
import torch
from config import Config
from model import VIT_B16, VIT_B32
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
configs = [Config('best'), Config('best2'),Config('best6')]
models = []
predictions = []
for config in configs:
    if config.model_name == 'VIT_B16':   
        model = VIT_B16(config=config).to(device)
    elif config.model_name == 'VIT_B32':
        model = VIT_B32(config=config).to(device)
    state = torch.load(config.ckpt_path, map_location=torch.device(device))
    model.load_state_dict(state['state_dict'])
    state = {'state_dict': model.state_dict()}
    torch.save(state, config.ckpt_path,_use_new_zipfile_serialization=False) 