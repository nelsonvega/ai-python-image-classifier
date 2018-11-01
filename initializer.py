 
import torch
import argparse

from torchvision import datasets,transforms,models

def init(root_dir,stages=['train', 'valid', 'test'],train_stage='train'):



    data_dir = root_dir
    train_dir = data_dir + '/'+stages[0]
    valid_dir = data_dir + '/'+stages[1]
    test_dir = data_dir + '/'+stages[2]

    dirs = {stages[0]: train_dir, 
            stages[1]: valid_dir, 
            stages[2] : test_dir}


    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms  = {
        stages[0]: transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])
        ]),
        stages[1]: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])
        ]),
        stages[2]: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])
        ]),
    }
 
 # TODO: Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in stages}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in stages}

    dataset_sizes = {x: len(image_datasets[x]) 
                                    for x in stages}

    class_names = image_datasets[train_stage].classes

    return image_datasets,dataloaders,dataset_sizes,class_names
            
def init_train_cmd_arguments():
    


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--hidden_units', type=int)
    parser.add_argument('--checkpoint_name', type=str)
    parser.add_argument('--root_dir', type=str)


    args, _ = parser.parse_known_args()

    return args