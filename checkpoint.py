import json
import time
import copy

import seaborn as sns
import torch.optim as optim

import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torchvision import datasets,transforms,models

from collections import OrderedDict

 
def load_checkpoint(checkpoint_name='ic-model.pth',gpu="True"):
   
   # loading the checkpoint to the local storage, either the CPU or the GPU
    if(gpu):
        checkpoint = torch.load(checkpoint_name)
    else:
        checkpoint = torch.load(checkpoint_name, map_location=lambda storage, loc: storage)

    hidden_units = checkpoint['hidden_units']
    print('loading the latest version')   
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    
    #reading values from the checkpoint

    #loading the actual model based on the saved architecture
    model.classifier = checkpoint['classifier']
    model.optimizer =   optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001, momentum=0.9)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def save_checkpoint(model,checkpoint_name='ic-model.pth',arch='vgg16',hidden_units=4096,class_to_idx={}):
    # TODO: Save the checkpoint 
# Save a checkpoint 
    model.class_to_idx =class_to_idx

    checkpoint = {
        'arch': arch,
        'class_to_idx': model.class_to_idx, 
        'state_dict': model.state_dict(),
        'hidden_units': hidden_units,
        'classifier':model.classifier,
       # 'optimizer':model.optimizer   
    }

    torch.save(checkpoint, checkpoint_name)


def get_inputsize(arch):
    if(arch == 'vgg19'):
         input_size=25088
    elif(arch =="alexnet"):
        input_size=9216
    elif(arch =='vgg16'):
        input_size=25088
    elif(arch =='squeezenet'):
        input_size=21725184
    else:
         input_size=25088
    return input_size

if __name__=="__main__":
   model=load_checkpoint(gpu=False)
   print (model)