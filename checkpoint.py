import json
import time
import copy

import seaborn as sns

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
    if(not (checkpoint.get('classifier') is None)):

        model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
        
        #reading values from the checkpoint

        #loading the actual model based on the saved architecture
        model.classifier = checkpoint['classifier']
        #model.optimizer = checkpoint('optimizer')


    else:
        arch = checkpoint['arch']

        #loading the actual model based on the saved architecture

        if(arch == 'VGG'):
            print('vgg selected')
            model = models.vgg19(pretrained=True)
        elif(arch =="alexnet"):
            model = models.alexnet(pretrained=True)
        elif(arch =='vgg16'):
            model = models.vgg16(pretrained=True)
        elif(arch =='squeezenet'):
            model = models.squeezenet1_0(pretrained=True)
        elif(arch =='densenet161'):
            model = models.densenet161(pretrained=True)
        else:
            model = models.inception_v3(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        input_size=get_inputsize(arch)

    #  Create the classifier
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(input_size, hidden_units)),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))        
        #Assign values
        model.classifier=classifier

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