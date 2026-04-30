from importlib import import_module
import os
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        module = import_module('model.'+args.model)
        self.model = module.make_model(args).to(self.device)

    def forward(self, x):
        return self.model(x)
    
    def get_model(self):
        return self.model
