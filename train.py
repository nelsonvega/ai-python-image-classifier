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


# Define command line argumentsparser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to dataset ')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')

args, _ = parser.parse_known_args()


def train_model(image_datasets,dataloaders, arch='vgg19', hidden_units=4096, 
                num_epochs=25, learning_rate=0.001, device='cpu'):
    
    # TODO: Build and train your network

    
    model = models.vgg19(pretrained=True)
    
    # Criteria NLLLoss which is recommended with Softmax final layer
    criterion = nn.NLLLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = classifier   
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
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

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    dirs = {'train': train_dir, 
            'valid': valid_dir, 
            'test' : test_dir}
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms  = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }


    # TODO: Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) 
                                  for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    
    print (dataset_sizes)
    print (class_names)
    
    import json

    with open('cat_to_name.json', 'r') as f:
        label_mapper = json.load(f)
            
    
    # Number of epochs
    eps=25
    
    model_ft = train_model(image_datasets=image_datasets,dataloaders=dataloaders, arch='vgg19', hidden_units=4096, 
                num_epochs=25, learning_rate=0.001, device='cuda')
    
    