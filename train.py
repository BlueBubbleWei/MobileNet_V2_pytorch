import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from cifar10data import CIFAR10Data
from utils import parse_args, create_experiment_dirs , AverageTracker
from model import MobileNet2

def train(model ,trainloader , testloader , epochs ,device, config_args):
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=device)

    optimizer = optim.SGD(model.parameters(), lr=config_args.learning_rate, momentum=config_args.momentum ,  weight_decay=config_args.weight_decay)

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config_args.step_size, gamma=config_args.gamma)
    
    for epoch in range(epochs):
        running_loss = 0.0 
        exp_lr_scheduler.step()
        
        for i , data in enumerate(trainloader,0):
            inputs ,labels = data
            inputs ,labels = inputs.to(device) , labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs ,labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                cur_acc1, cur_acc5 = compute_accuracy(outputs.data, labels, topk=(1,5))
                print('[%d, %5d] loss: %.3f  acc_1: %.4f  acc_5: %.4f' %(epoch + 1, i + 1, running_loss / 20 , cur_acc1 ,cur_acc5))
                running_loss = 0.0
                
       # test(model , testloader ,device)
                
        if epoch % 10==9 or epoch == epochs-1:
            torch.save(model, './models/model_%d.pkl'%(epoch+1))
            print('Saved model %d epoch'%(epoch+1))

def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, idx = output.topk(maxk, 1, True, True)
    idx = idx.t()
    correct = idx.eq(target.view(1, -1).expand_as(idx))

    acc_arr = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc_arr.append(correct_k.mul_(1.0 / batch_size))
    return acc_arr

def test(model , testloader , device):
    top1, top5 = AverageTracker(), AverageTracker()

    with torch.no_grad():
    
        for i , data in enumerate(testloader,0):
            inputs ,labels = data
            inputs ,labels = inputs.to(device) , labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            cur_acc1, cur_acc5 = compute_accuracy(outputs.data, labels, topk=(1,5))
            
            top1.update(cur_acc1[0])
            top5.update(cur_acc5[0])

        print("Test Results" + " | " + " acc-top1: " + str(top1.avg) + "acc-top5: " + str(top5.avg))

# 保存和加载整个模型
#torch.save(model_object, 'model.pkl')
#model = torch.load('model.pkl')

# 仅保存和加载模型参数
#torch.save(model_object.state_dict(), 'params.pkl')
#model_object.load_state_dict(torch.load('params.pkl'))

