import torch

def calc_accuracy(original, predicted):
    # ref: https://pytorch.org/docs/stable/torch.html#module-torch
    n, c = original.shape
    div = n*c

    return torch.round(predicted).eq(original).sum().cpu().detach().numpy() / div