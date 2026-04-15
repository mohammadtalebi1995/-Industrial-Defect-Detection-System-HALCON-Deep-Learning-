import torch
from model import Autoencoder

# load dataset (MNIST normal only)
# train loop (same as you implemented)

# save model
torch.save(model.state_dict(), "model.pth")