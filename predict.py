

import json
import time
import copy
import checkpoint as loader
import argparse

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

from torch  import nn,optim
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import datasets,transforms,models

from collections import OrderedDict
import torch.optim as optim
from torch.optim import lr_scheduler


def imshow(image, ax=None, title=None):
 
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    #image_pil=Image.open(image_path)

    loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    image_pl = Image.open(image_path)
    imagepl_ft = loader(image_pl).float()
    
    #cropped_size=256,256
    
    #image_pil.thumbnail(cropped_size)
    
    #left_margin = (image_pil.width-224)/2
    #bottom_margin = (image_pil.height-224)/2
    #right_margin = left_margin + 224
    #top_margin = bottom_margin + 224
    #image_pil = image_pil.crop((left_margin, bottom_margin, right_margin,
    #                 top_margin))
   
    
    np_image=np.array(imagepl_ft)
    
    #np_image=np_image/255
    
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    #np_image = (np_image - mean)/std
    
    #np_image=np_image.transpose((2,0,1))

    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))

    
    return np_image

def predict(image_path, model_name, topk=10, categories='', gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file

    with open('cat_to_name.json', 'r') as f:
        label_mapper = json.load(f)

    model=loader.load_checkpoint(model_name,gpu=gpu)

    img=process_image(image_path)
    
    img=torch.from_numpy(img).type(torch.FloatTensor)
    
    inpt=img.unsqueeze(0)
    
    model_result=model.forward(inpt)
    
    expResult=torch.exp(model_result)
    
    firstTopX,SecondTopX=expResult.topk(topk)
    
    probs = torch.nn.functional.softmax(firstTopX.data, dim=1).numpy()[0]
    #classes = SecondTopX.data.numpy()[0]

    #probs = firstTopX.detach().numpy().tolist()[0] 
    classes = SecondTopX.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    #labels = [label_mapper[str(lab)] for lab in SecondTopX]
    labels  = [idx_to_class[y] for y in classes]
    flowers=[categories[idx_to_class[i]] for i in classes]
  
    return probs,flowers



def show_prediction(image_path,probabilities,labels, categories):

    plt.figure(figsize=(6,10))
    ax=plt.subplot(2,1,1)
    
    flower_index=image_path.split('/')[2]
    name=categories[flower_index]
    img=process_image(image_path)
    imshow(img,ax)
    
    plt.subplot(2,1,2)
    sns.barplot(x=probabilities,y=labels,color=sns.color_palette()[0])
    plt.show()



if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--category_names', type=str)
    parser.add_argument('--top_k', type=int)


    args, _ = parser.parse_known_args()

    if (args.input):
        input_name=args.input
    else:
        input_name='flowers/test/28/image_05230.jpg'

    if(args.checkpoint):
        checkpoint=args.checkpoint
    else:
        checkpoint='ic-model.pth'

    if(args.category_names):
        category_names=args.category_names
    else:
        category_names='cat_to_name.json'

    if(args.gpu):
            device='cuda'
    else:
        device='cpu'

   # show_prediction(image_path=input_name,model=checkpoint,category_names=category_names)

   
    with open(category_names, 'r') as f:
        categories = json.load(f)

    # run the prediction
    probabilities,labels=predict(input_name,checkpoint,topk=5,categories=categories,gpu=device)

    # show prediction
    print('predict results')
    print(probabilities)
    print(labels)
    show_prediction(image_path=input_name,probabilities=probabilities,labels= labels, categories= categories)





