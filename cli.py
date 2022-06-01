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
model_state = trained_model['model_state']


model = Classifier_Model_1(28*28)
model.load_state_dict(model_state)

parser = argparse.ArgumentParser(description= 'Clothing image classifier')

parser.add_argument('test_data_dir', type=str, help='Input path to test data')

args = parser.parse_args()

test_data_root = args.test_data_dir


transform_test   = transforms.Compose([  transforms.Grayscale(), transforms.Resize((28, 28)),  transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])  ])
# dataset_test     = datasets.FashionMNIST(root=test_data_root, download=True, train=False, transform=transform_test)
# dataloader_test  = DataLoader(dataset=dataset_test,  batch_size=64, shuffle=True)

# data_iter = iter(dataloader_test)
# images, labels = data_iter.next()


# check_image = images[1]
# check_image_flatten = check_image.view(check_image.shape[0], -1)
# prediction = model.forward(check_image_flatten)


# view_classify(check_image, prediction, version='Fashion')
# plt.show()


processed_image = datasets.ImageFolder(root=test_data_root, transform=transform_test)
dataloader_test  = DataLoader(dataset=processed_image,  batch_size=2, shuffle=True)

data_iter = iter(dataloader_test)
images, labels = data_iter.next()



check_image = images[0]

check_image_flatten = check_image.view(check_image.shape[0], -1)
prediction = model.forward(check_image_flatten)


view_classify(check_image, prediction, version='Fashion')
plt.show()