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

class MultiTaskLoss_cls(nn.Module):
    def __init__(self, is_mse=True):
        super().__init__()
        self.is_mse = is_mse

        self.ce = nn.CrossEntropyLoss()
        self.mate_bce_w_logits = nn.BCEWithLogitsLoss()
        self.tech_bce_w_logits = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, target, mate, tech):
        target_out, mate_out, tech_out = outputs

        loss = self.ce(target_out, target.long()) + self.mate_bce_w_logits(mate_out, mate) + self.tech_bce_w_logits(tech_out, tech)

        return loss

class MultiTaskLoss_v2(nn.Module):
    def __init__(self, is_mse=True):
        super().__init__()
        self.is_mse = is_mse

        if self.is_mse:
            self.mse1 = nn.MSELoss()
            self.mse2 = nn.MSELoss()
        else:
            self.mse1 = nn.SmoothL1Loss()
            self.mse2 = nn.SmoothL1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.mate_bce_w_logits = nn.BCEWithLogitsLoss()
        self.tech_bce_w_logits = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, target_hard, target_ce, target_soft, mate, tech):
        target_hard_out, ce_out, target_soft_out, mate_out, tech_out = outputs

        #loss = self.mse(target_hard_out, target_hard) + self.ce(ce_out, target_ce) + self.smooth_l1(target_soft_out, target_soft) + self.mate_bce_w_logits(mate_out, mate) + self.tech_bce_w_logits(tech_out, tech)
        loss = self.mse1(target_hard_out, target_hard) + self.ce(ce_out, target_ce.long()) + self.mse2(target_soft_out, target_soft) + self.mate_bce_w_logits(mate_out, mate) + self.tech_bce_w_logits(tech_out, tech)

        return loss

class MultiTaskLoss_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

        self.mate_bce_w_logits = nn.BCEWithLogitsLoss()
        self.tech_bce_w_logits = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, target, target_logits, mate, tech):
        target_out, logits_out, mate_out, tech_out = outputs

        loss = self.mse(target_out, target) + self.ce(logits_out, target_logits.long()) + self.mate_bce_w_logits(mate_out, mate) + self.tech_bce_w_logits(tech_out, tech)

        return loss