import torch
import torch.nn as nn

# Define a sample model
model = nn.Linear(3, 1)
model.train()
input_tensor = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
target = torch.tensor([[10.0]])

# Forward pass
output = model(input_tensor)
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)

# Backward pass
loss.backward()

# Check gradients
print("Gradients of the input tensor:", input_tensor.grad)
print("Gradients of the model weights:")
print("Weight:", model.weight.grad)
print("Bias:", model.bias.grad)
