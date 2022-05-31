from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
from torch.utils.data import DataLoader, Dataset
import data_handler as dh

with st.sidebar:
    menu = option_menu(menu_title='Contents',
                        menu_icon='menu-up',
                        options=['Home', 'Image classifier'],
                        icons=['house', 'aspect-ratio'],
                        orientation='vertical')

if menu == 'Home':
    st.title('Clothing image classifier')
    st.markdown("Created by: _Peter, Sardorbek_")
    image_1 = Image.open("Ralph-Lauren-Logo.png")
    st.image(image_1)
    st.markdown('We have used the Fashion MNIST dataset to create a Neural Network that can classify clothing types in this project.')
    st.markdown('The dataset consists of 60,000 28x28 Greyscale images for training and 10,000 images for test sets. ')
    image_2 = Image.open('unnamed-chunk-8-1.png')
    st.image(image_2)
    st.markdown('As the project included some extra challenges, we have tried to create a model with high accuracy and that can classify other normal clothing images.')

else:
    st.title('Clothing classifier model')
    st.markdown('Please upload an image')
    uploaded_file = st.file_uploader('Choose a clothing photo', accept_multiple_files=False)
