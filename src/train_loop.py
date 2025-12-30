import torch
import torch.nn as nn
import time
import math

def train_epoch(model, loader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(loader):
        src = batch["code_ids"].to(device)
        trg = batch["summary_ids"].to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        # trg: [batch size, trg len]
        # output: [batch size, trg len, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        # trg = [ (trg len - 1) * batch size ]
        # output = [ (trg len - 1) * batch size, output dim ]
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if i % 100 == 0:
             print(f"Batch {i}/{len(loader)} Loss: {loss.item():.4f}")
        
    return epoch_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            src = batch["code_ids"].to(device)
            trg = batch["summary_ids"].to(device)
            
            output = model(src, trg, 0) # turn off teacher forcing
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(loader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
