import torch
from torch import nn, optim

from model import DModel

from utils import train, train_custom_model

model = DModel("model.yaml")

data = "./dataset/data.yaml"

train_custom_model(model, data, nn.CrossEntropyLoss(), optim.SGD(model.parameters(), lr=0.001), epochs=30)
torch.save(model.state_dict(), "/tmp/model.pt")
