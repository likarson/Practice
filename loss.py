# define prediction and target tensors
predictions = torch.tensor([-6.9229, -29.8163, -16.0748, -13.2427, -14.1096], dtype=torch.float)
y = torch.tensor([2550, 11500, 3000, 4500, 4795], dtype=torch.float)

## YOUR SOLUTION HERE ##
loss = nn.MSELoss()
MSE = loss(predictions, y)
# show output
print("MSE Loss:", MSE)

RMSE = MSE**(1/2)

# show output
RMSE
