# -*- coding:utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
import glob
from cifar10data import CIFAR10Data
from utils import parse_args
from model import MobileNet2
from train import train , test
from cifar10data import CIFAR10_LABELS_LIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

test_transform = transforms.Compose(
    [transforms.Resize((32,32)) ,
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.49139967861519607, 0.48215840839460783, 0.44653091444546567], std=[0.24703223246328262, 0.2434851280000555, 0.26158784172796473]),
    ])

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = test_transform(image).unsqueeze_(0)   ## batch dimension
    return image.to(device, torch.float)
                
def main():
    # Parse the JSON arguments
    config_args = parse_args()
    
    if config_args.run_mode == "train" :
        model = MobileNet2(scale=1.0, input_size=config_args.input_size, t=6, in_channels=config_args.num_channels, num_classes=config_args.num_classes)
        model = model.to(device=device)

        num_parameters = sum([l.nelement() for l in model.parameters()])
        print(model)
        print('number of parameters: {}'.format(num_parameters))

        print("Loading Data...")
        data = CIFAR10Data(config_args)
        print("Data loaded successfully\n")

        train(model , data.trainloader , data.testloader , config_args.num_epochs , device ,config_args)

        print('Finished Training')
        
    else :
        model = torch.load('models/model_800.pkl')
        model = model.eval()
        model = model.to(device)
        
       # test_image = image_loader("images/14.jpg")
        image_list=glob.glob('./images/*.jpg')
        
        for img_str in image_list:
            test_image = image_loader(img_str)
            output = model(test_image)
            index = torch.max(output, 1, keepdim=False, out=None)
            print("Image %s is a : %s " %(img_str[9:],CIFAR10_LABELS_LIST[index[1]]))
 
if __name__ == "__main__":
    main()
    
    
    
    
    
