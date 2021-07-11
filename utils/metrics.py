import torch

def calc_accuracy(original, predicted):
    # ref: https://pytorch.org/docs/stable/torch.html#module-torch
    return torch.round(predicted).eq(original).sum().cpu().detach().numpy() / len(original)