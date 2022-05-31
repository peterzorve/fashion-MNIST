import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
from torch.utils.data import DataLoader, Dataset

# load the data
def load_data():
    transform_train = transforms.Compose([  transforms.Resize((28, 28)), 
                                            transforms.RandomHorizontalFlip(p=0.4), 
                                            transforms.RandomRotation(20), 
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5], std=[0.5])
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])

    transform_test  = transforms.Compose([  transforms.Resize((28, 28)), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize(mean=[0.5], std=[0.5])
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])


    dataset_train = datasets.FashionMNIST(root='data', download=True, train=True, transform=transform_train)
    dataset_test  = datasets.FashionMNIST(root='data', download=True, train=False, transform=transform_test)


    dataloader_train = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
    dataloader_test  = DataLoader(dataset=dataset_test,  batch_size=64, shuffle=True)
    return dataset_train, dataset_test, dataloader_test, dataloader_train