

from numpy import average
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms 
import torch.optim as optim 
from model import Classifier_Model_1, Classifier_Model_2
import matplotlib.pyplot as plt 


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

model = Classifier_Model_1(784)

optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()


epochs, all_train_losses, all_test_losses = 10, [], []

for epoch in range(epochs):
     training_loss = 0
     for features_train, target_train in iter(dataloader_train):
          print()
          print(features_train.shape)
          print()

          features_train = features_train.view(features_train.shape[0], -1)
          # features_train.resize_(features_train.size()[0], 28*28)

          print()
          print(features_train.shape)
          print()

          optimizer.zero_grad()
          prediction_train = model.forward(features_train)
          loss_train = criterion(prediction_train, target_train)
          loss_train.backward()
          optimizer.step()

          training_loss += loss_train.item()

     average_training_loss = training_loss/len(dataloader_train)
     all_train_losses.append(average_training_loss)

     model.eval()
     testing_loss = 0 
     with torch.no_grad():
          for features_test, targets_test in iter(dataloader_test):

               features_test = features_test.view(-1, 28*28)

               prediction_test = model.forward(features_test)
               loss_test = criterion(prediction_test, targets_test)
               testing_loss += loss_test.item()
          
          average_testing_loss = testing_loss/len(dataloader_test)
          all_test_losses.append(average_testing_loss)
     model.train()


     print(f'{epoch:3}/{epochs}  |  Training Loss  :  {average_training_loss:.8f}   |   Testing Loss  :  {average_testing_loss:.8f}')


torch.save(model.state_dict(), 'trained_model')

plt.plot(all_train_losses, label='Train Loss')
plt.plot(all_test_losses,  label='Test Loss')
plt.legend()
plt.show()





          


