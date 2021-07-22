import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, is_mse=True):
        super().__init__()
        self.is_mse = is_mse

        if self.is_mse:
            self.mse = nn.MSELoss()
        else:
            self.mse = nn.SmoothL1Loss()
        self.mate_bce_w_logits = nn.BCEWithLogitsLoss()
        self.tech_bce_w_logits = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, target, mate, tech):
        target_out, mate_out, tech_out = outputs

        loss = self.mse(target_out, target) + self.mate_bce_w_logits(mate_out, mate) + self.tech_bce_w_logits(tech_out, tech)

        return loss