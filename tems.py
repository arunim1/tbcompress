import torch

# load inputs and outputs
inputs = torch.load("inputs.pt")
outputs = torch.load("outputs.pt")

print(inputs.shape)
print(outputs.shape)
