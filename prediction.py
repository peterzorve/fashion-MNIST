
from tabnanny import check
import torch 
import torch.nn as nn 
from model import Classifier_Model_1, Classifier_Model_2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 


from functions import imshow, view_classify 





trained_model = torch.load('trained_model')

model_state = trained_model['model_state']

model = Classifier_Model_1(28*28)
model.load_state_dict(model_state)



transform_test   = transforms.Compose([  transforms.Grayscale(), transforms.Resize((28, 28)),  transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])  ])
dataset_test     = datasets.FashionMNIST(root='data_validate', download=True, train=False, transform=transform_test)
dataloader_test  = DataLoader(dataset=dataset_test,  batch_size=64, shuffle=True)

data_iter = iter(dataloader_test)
images, labels = data_iter.next()


check_image = images[1]
check_image_flatten = check_image.view(check_image.shape[0], -1)
prediction = model.forward(check_image_flatten)


view_classify(check_image, prediction, version='Fashion')
plt.show()




###############################################          REAL DATA    ##################################################

# processed_image = datasets.ImageFolder(root="C:/Users/Omistaja/Desktop/image_folder", transform=transform_test)
# dataloader_test  = DataLoader(dataset=processed_image,  batch_size=2, shuffle=True)

# data_iter = iter(dataloader_test)
# images, labels = data_iter.next()



# check_image = images[0]

# check_image_flatten = check_image.view(check_image.shape[0], -1)
# prediction = model.forward(check_image_flatten)


# view_classify(check_image, prediction, version='Fashion')
# plt.show()

#########################################################################################################################


































# from PIL import Image 

# image1 = Image.open("C:/Users/Omistaja/Desktop/image_folder/Dress/dress_1.jpg")
# image_preprocess =  transforms.Compose([  transforms.Grayscale(), transforms.Resize((28, 28))  ])
# image1 = image_preprocess(image1)

# image2 = transform_test(image1)



# image2 = image2.view(image2.shape[0], -1)
# print(image2.shape)

# prediction = model.forward(image2)


# view_classify(image1, prediction, version='Fashion')
# plt.show()



