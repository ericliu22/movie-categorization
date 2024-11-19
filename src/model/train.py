import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .model import CategorizationModel


def train(current_epoch: int, model: CategorizationModel, optimizer: Optimizer, data_loaders: dict[str, DataLoader], device: str = "cpu"):

    LOG_INTERVAL: int = 100

    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for batch, (input, target) in enumerate(data_loaders['train']):
        
        optimizer.zero_grad()
        input, target = input.to(device), target.to(device)
        output = model(input)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        iteration: int = batch * len(input)
        dataset_len: int = len(data_loaders["train"].dataset)

        if batch % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(current_epoch, iteration, dataset_len,
            100. * batch / dataset_len, loss.item()))
            #Optionally save the model to continue training later

            # torch.save({
            #     'epoch': current_epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }, "checkpoint.pth")
