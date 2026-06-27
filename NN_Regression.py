import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# set a random seed - do not modify
torch.manual_seed(42)

## create the NN_Regression class
class NN_Regression(nn.Module):
    def __init__(self):
        super(NN_Regression, self).__init__()
        # initialize layers
        self.layer1 = nn.Linear(3, 16)
        self.layer2 = nn.Linear(16, 8) 
        self.layer3 = nn.Linear(8, 4)
        self.layer4 = nn.Linear(4, 1) 
        
        # initialize activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # define the forward pass
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x

## create an instance of NN_Regression
model = NN_Regression()

## create an input tensor

apartments_df = pd.read_csv("streeteasy.csv")
numerical_features = ['size_sqft', 'bedrooms', 'building_age_yrs']
X = torch.tensor(apartments_df[numerical_features].values, dtype=torch.float)

## feedforward to predict rent
predicted_rents = model(X)

## show output
predicted_rents[:5]


# set a random seed - do not modify
torch.manual_seed(42)

## create the NN_Regression class
class OneHidden(nn.Module):
    def __init__(self):
        super(OneHidden, self).__init__()
        # initialize layers
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        
        # initialize activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        ## YOUR SOLUTION HERE ##
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
     
        return x

## do not modify below this comment

# create an instance
model = OneHidden()

# create an input tensor
X = torch.tensor([3,4.5], dtype=torch.float32)

# run feedforward
predictions = model(X)

# show output
predictions


# set a random seed - do not modify
torch.manual_seed(42)

## create the NN_Regression class
class OneHidden(nn.Module):
    # add a new numHiddenNodes input
    def __init__(self, numHiddenNodes):
        super(OneHidden, self).__init__()
        # initialize layers
        # 3 input features, variable output features
        self.layer1 = nn.Linear(2, numHiddenNodes) 
        # variable input features, 8 output features
        self.layer2 = nn.Linear(numHiddenNodes, 1) 
        
        # initialize activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        ## YOUR SOLUTION HERE ##
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

## YOUR SOLUTION HERE ##
model = OneHidden(10)

## do not modify below this comment

# create an input tensor
input_tensor = torch.tensor([3,4.5], dtype=torch.float32)

# run feedforward
predictions = model(input_tensor)

# show output
predictions
