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

 
def load_checkpoint(check_point='ic-model.pth'):
   
   # loading the checkpoint to the local storage, either the CPU or the GPU
    checkpoint = torch.load(check_point, map_location=lambda storage, loc: storage)

    #reading values from the checkpoint
    arch = checkpoint['arch']
    num_labels = len(checkpoint['class_to_idx'])
    hidden_units = checkpoint['hidden_units']
    state_dict=checkpoint['state_dict']
    
    #loading the actual model based on the saved architecture
    if(arch == 'vgg19'):
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
    
    model.class_to_idx = num_labels
            
   #  Create the classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))        
    #Assign values
    model.classifier=classifier
    model.load_state_dict(state_dict)
    
    
    return model

    
def save_checkpoint(model,checkpoint_name='ic-model.pth',hidden_units=4096,class_to_idx=102):
    # TODO: Save the checkpoint 
# Save a checkpoint 
    model.class_to_idx =class_to_idx

    checkpoint = {
        'arch': 'VGG',
        'class_to_idx': model.class_to_idx, 
        'state_dict': model.state_dict(),
        'hidden_units': hidden_units    
    }

    torch.save(checkpoint, checkpoint_name)

if __name__=="__main__":
   model=load_checkpoint()
   print (model)