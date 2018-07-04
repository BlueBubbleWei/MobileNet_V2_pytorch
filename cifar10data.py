import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import calc_dataset_stats

# Example DataLoader on CIFAR-10

class CIFAR10Data:
    def __init__(self, args):
        mean, std = calc_dataset_stats(torchvision.datasets.CIFAR10(root='/home/zzg/DeepLearning/Pytorch/data', train=True,
                                                                    download=args.download_dataset).train_data,
                                       axis=(0, 1, 2))
                                       
#mean  [0.49139967861519607, 0.48215840839460783, 0.44653091444546567]
#std   [0.24703223246328262, 0.2434851280000555, 0.26158784172796473]

        train_transform = transforms.Compose(
            [transforms.RandomCrop(args.input_size),
             transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(0.3, 0.3, 0.3),
             transforms.ToTensor(),                 # range [0, 255] -> [0.0,1.0]  [C, H, W]
             transforms.Normalize(mean=mean, std=std)])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.49139967861519607, 0.48215840839460783, 0.44653091444546567], std=[0.24703223246328262, 0.2434851280000555, 0.26158784172796473])])

        self.trainloader = DataLoader(torchvision.datasets.CIFAR10(root='/home/zzg/DeepLearning/Pytorch/data', train=True,
                                                                   download=args.download_dataset,
                                                                   transform=train_transform),
                                      batch_size=args.batch_size,
                                      shuffle=args.shuffle, num_workers=args.dataloader_workers,
                                      pin_memory=args.pin_memory)

        self.testloader = DataLoader(torchvision.datasets.CIFAR10(root='/home/zzg/DeepLearning/Pytorch/data', train=False,
                                                                  download=args.download_dataset,
                                                                  transform=test_transform),
                                     batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.dataloader_workers,
                                     pin_memory=args.pin_memory)


CIFAR10_LABELS_LIST = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
