# run training loop

import torch
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        
        # move images and masks to `device` (GPU)
        images, masks = images.to(device), masks.to(device)
        
        # zero the parameter gradients (optimizer.zero_grad())
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(images)
        
        # calculate loss 
        loss = criterion(outputs, masks)
        
        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
        
        # sum loss
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad(): 
        for images, masks in tqdm(dataloader, desc="Validating"):
            
            # move to device
            images, masks = images.to(device), masks.to(device)

            # forward pass
            outputs = model(images)

            # calculate loss
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # calculate dice score
            dice_score = 1.0 - loss.item()
            val_dice += dice_score
            
    return val_loss / len(dataloader), val_dice / len(dataloader)