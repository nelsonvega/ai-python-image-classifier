# Imports here

import json
import time
import copy

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from torch  import nn,optim
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torchvision import datasets,transforms,models

from collections import OrderedDict
import torch.optim as optim
from torch.optim import lr_scheduler

 
def load_checkpoint(check_point='ic-model1.pth',arch='vgg19'):
   
    checkpoint = torch.load(check_point, map_location=lambda storage, loc: storage)
    arch = checkpoint['arch']
    
    num_labels = len(checkpoint['class_to_idx'])
    hidden_units = checkpoint['hidden_units']
    state_dict=checkpoint['state_dict']
    
    if(arch == 'vgg19'):
        model = models.vgg19(pretrained=True)
    elif(arch =="alexnet"):
        model = models.alexnet(pretrained=True)
    elif(arch =='vgg16'):
        vgg16 = models.vgg16(pretrained=True)
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
    model.classifier=classifier
    model.load_state_dict(state_dict)
    
    
    return model