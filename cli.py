import argparse
import joblib
import torch 
import torch.nn as nn 
from backup_from_peter.model_s import Classifier_Model_1, Classifier_Model_2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 


from backup_from_peter.functions_s import imshow, view_classify 
trained_model = torch.load('backup_from_peter/trained_model_S')

model = Classifier_Model_1(28*28)
model.load_state_dict(model_state)