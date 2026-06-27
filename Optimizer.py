import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import torch.optim as optim

# create neural network
model = nn.Sequential(
    nn.Linear(3,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4,1)
)

# import the data
apartments_df = pd.read_csv("streeteasy.csv")
numerical_features = ['bedrooms', 'bathrooms', 'size_sqft']
X = torch.tensor(apartments_df[numerical_features].values, dtype=torch.float)
y = torch.tensor(apartments_df['rent'].values,dtype=torch.float)

# forward pass
predictions = model(X)

# define the loss function and compute loss
loss = nn.MSELoss()
MSE = loss(predictions,y)
print('Initial loss is ' + str(MSE))

## ##
optimizer = optim.Adam(model.parameters(), lr = 0.001)
MSE.backward()
optimizer.step()

# feed the data through the updated model and compute the new loss
predictions = model(X)
MSE = loss(predictions,y)
print('After optimization, loss is ' + str(MSE))

