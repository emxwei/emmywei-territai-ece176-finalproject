# dice loss implementation
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        # apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # calculate intersection
        intersection = (inputs * targets).sum()

        # calculate denominator
        denominator = inputs.sum() + targets.sum()
        
        # calculate dice value
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        
        # calculate dice loss
        loss = 1 - dice

        return loss