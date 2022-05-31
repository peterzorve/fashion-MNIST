

from turtle import forward
import torch 
import torch.nn as nn

class Classifier_Model_1(nn.Module):
     def __init__(self, input):

          super(Classifier_Model_1, self).__init__()

          self.input_layer = nn.Linear(input, 64)
          self.first_hidden = nn.Linear(64, 64)
          self.second_hidden = nn.Linear(64, 64)
          self.third_hidden = nn.Linear(64, 64)
          self.output_layer = nn.Linear(64, 10)

          self.sigmoid = nn.Sigmoid()
          self.softmax = nn.Softmax(dim=1)

     def forward(self, x):
          x = self.sigmoid(  self.input_layer(x) )
          x = self.sigmoid(  self.first_hidden(x) )
          x = self.sigmoid(  self.second_hidden(x) )
          x = self.sigmoid(  self.third_hidden(x) )
          x = self.softmax(  self.output_layer(x) )

          return x 


class Classifier_Model_2(nn.Module):
     def __init__(self, input):
          super(Classifier_Model_2, self).__init__()

          self.first_later = nn.Linear(input, 64)
          self.second_layer = nn.Linear(64, 64)
          self.third_layer = nn.Linear(64, 32)
          self.last_layer = nn.Linear(32, 10)

          self.relu = nn.ReLU()
          self.sigmoid = nn.Sigmoid()
          self.softmax = nn.Softmax(dim=1)

     def forward(self, x):
          x = self.relu( self.first_later(x) )
          x = self.relu( self.second_layer(x) )
          x = self.relu( self.third_layer(x) )
          x = self.softmax( self.last_layer(x) )

          return x 