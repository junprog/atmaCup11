import torch

def calc_accuracy(predicted, target):
    # ref: https://pytorch.org/docs/stable/torch.html#module-torch
    n, c =  target.shape
    div = n*c

    return torch.round(predicted).eq(target).sum().cpu().detach().numpy() / div

def calc_accuracy_ce(predicted, target):
    # ref: https://pytorch.org/docs/stable/torch.html#module-torch
    div = target.shape
    
    acc = torch.sum(torch.argmax(predicted, dim=1) == target)

    return float(acc.cpu().detach().numpy() / div)