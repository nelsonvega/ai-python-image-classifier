import json
import time
import copy

import initializer
import checkpoint as loader
import argparse

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from torch  import nn
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torchvision import datasets,transforms,models

from collections import OrderedDict
import torch.optim as optim
from torch.optim import lr_scheduler



def train_model(image_datasets,dataloaders,dataset_sizes, arch='vgg19', hidden_units=4096, 
                num_epochs=25, learning_rate=0.001,dropout=0.5, device='cpu'):
    
    # TODO: Build and train your network

    if(not torch.cuda.is_available() and device=='cuda'):
        device='cpu'

    input_size=25088
    if(arch == 'vgg19'):
        model = models.vgg19(pretrained=True)
    elif(arch =="alexnet"):
        model = models.alexnet(pretrained=True)
        input_size=9216
    elif(arch =='vgg16'):
        model = models.vgg16(pretrained=True)
    elif(arch =='squeezenet'):
        model = models.squeezenet1_0(pretrained=True)
        input_size=21725184
    elif(arch =='densenet161'):
        model = models.densenet161(pretrained=True)
    print(arch)
    print(model)
    # Features, removing the last layer
    print('Architecture:'+arch+' Input size:'+str(input_size)+ ' Device :' +device)
    

    # Extend the existing architecture with new layers
    classifier = nn.Sequential(OrderedDict([
                              ('dropout',nn.Dropout()),
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('relu', nn.ReLU(inplace=True)),
                              ('drop1', nn.Dropout()),
                              ('hidden_layer2',nn.Linear(4096,4096)),
                              ('relu4',nn.ReLU(inplace=True)),
                              ('fc2', nn.Linear(4096, 102))
                             # ('output', nn.LogSoftmax(dim=1))
                              ]))

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = classifier   
       
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    #optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
 
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    model.to(device)

    for epoch in range(num_epochs):
        print('Iteration {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train() 
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0
            print('Phase:'+phase)
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
               
              
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    #outputs = model.forward(inputs)
                    #preds = torch.exp(outputs).data
                    outputs = model(inputs)
                 
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
      

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)


    #if(save_dir and model_name):


    return model


if __name__=="__main__":

    args=initializer.init_train_cmd_arguments()

    image_datasets,dataloaders,dataset_sizes,class_names=initializer.init(root_dir="flowers",stages=['train','valid','test'],train_stage='train')

    if(args.epochs):
        eps=args.epochs
    else:
        eps=25
    if(args.learning_rate):
        learning_rate=args.learning_rate
    else:
        learning_rate=0.001

    if(args.hidden_units):
        hidden_units=args.hidden_units
    else:
        hidden_units=4096

    if(args.gpu):
        device='cuda'
    else:
        device='cpu'

    if(args.arch):
        architecture=args.arch
    else:
        architecture='vgg16'

    if(args.checkpoint_name):
        checkpoint_name=args.checkpoint_name
    else:
        checkpoint_name='ic-model.pth'

    if(args.root_dir):
        root_dir=args.root_dir
    else:
        root_dir='/'

    model = train_model(image_datasets=image_datasets,dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                arch=architecture, hidden_units=hidden_units, 
                num_epochs=eps, 
                learning_rate=learning_rate, device=device)

    print(model)
    class_to_idx=image_datasets['train'].class_to_idx
 
loader.save_checkpoint(model=model,checkpoint_name=checkpoint_name,arch=architecture,hidden_units=hidden_units,class_to_idx=class_to_idx,learning_rate=learning_rate)