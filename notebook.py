<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# # Import Libraries
# 
# Import PyTorch, pandas, NumPy, and scikit-learn. (Or feel free to import them as needed in the cells below.)

# In[2]:


import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


# # Import Data
# 
# Import the `streeteasy.csv` dataset and preview the first few rows.

# In[3]:


apartments_df = pd.read_csv(&#34;streeteasy.csv&#34;)
apartments_df.head()


# # Select Target
# 
# Select the numeric column that the neural network will be trying to predict. Feel free to use rent again, or try to predict another column!
# 
# Convert this column to a PyTorch tensor.

# In[5]:


numerical_features = [&#39;bedrooms&#39;, &#39;bathrooms&#39;, &#39;size_sqft&#39;, &#39;min_to_subway&#39;, &#39;floor&#39;, &#39;building_age_yrs&#39;,
                      &#39;no_fee&#39;, &#39;has_roofdeck&#39;, &#39;has_washer_dryer&#39;, &#39;has_doorman&#39;, &#39;has_elevator&#39;, &#39;has_dishwasher&#39;,
                      &#39;has_patio&#39;, &#39;has_gym&#39;]

X = torch.tensor(apartments_df[numerical_features].values, dtype=torch.float)
y = torch.tensor(apartments_df[&#39;rent&#39;].values, dtype=torch.float).view(-1,1)


# # Select Features
# 
# Select the numeric columns that the neural network will use as input features to predict the target.

# In[ ]:





# # Train-Test-Split
# 
# Split the features and target into training and testing datasets. A good initial proportion is 80/20.

# In[6]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(
    X,y,train_size = 0.70, test_size = 0.30,random_state = 2
)


# # Create a Neural Network
# 
# Create a neural network using either `Sequential` or OOP. Remember, the first `nn.Linear()` needs to match the number of input features, and the final output needs to have one node for regression.

# In[7]:


# set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(14, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)


# # Select a Loss Function
# 
# Select a loss function. Feel free to use MSE again, or check out PyTorch&#39;s other [loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions). A good alternate to MSE is `nn.L1Loss()`, which is the Mean Absolute Error.

# In[8]:


# MSE loss function + optimizer
loss = nn.MSELoss()


# # Select an Optimizer
# 
# Select an optimizer. Feel free to use Adam again, or check out PyTorch&#39;s other [optimizers](https://pytorch.org/docs/stable/optim.html#algorithms). A good alternate to Adam is `nn.SGD`, another gradient descent algorithm (stochastic gradient descent).

# In[9]:


optimizer = optim.Adam(model.parameters(), lr=0.001)


# # Training Loop
# 
# Use your selected loss and optimizer functions to train the neural network.

# In[10]:


num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X_train) 
    MSE = loss(predictions, y_train) 
    MSE.backward()
    optimizer.step() 
    optimizer.zero_grad()
    
    ## DO NOT MODIFY ##
    # keep track of the loss during training
    if (epoch + 1) % 100 == 0:
        print(f&#39;Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}&#39;)


# # Experiment
# 
# Go back and experiment with changing the setup of your neural network. Can you improve its performance using different activation functions or network architecture? What about adjusting the learning rate or switching out loss functions and optimizers?

# # Evaluate
# 
# As you experiment, evaluate each version of your model on the testing dataset, to validate its performance on unseen data.

# In[11]:


torch.save(model, &#39;model.pth&#39;)


# # Save the Final Network
# 
# Save your final network for later use.

# In[ ]:





# # Share Results
# 
# Share your network&#39;s performance and how you got there! Create a GitHub repo, or post in the Codecademy forums/discord.

# In[ ]:




<script type="text/javascript" src="/relay.js"></script></body></html>