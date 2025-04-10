import torch
from data import Data
from train import train
from model import RNN


data = Data()
model = RNN(len(data.vocab), 256, len(data.categories))
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)


train(
    model,
    criterion,
    optimizer,
    epochs=100000,
    data=data
)