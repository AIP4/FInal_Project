import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn

def compute_permutation_importance(model, val_loader, feature_columns, device, loss_fn=nn.L1Loss()):
    model.eval()
    
    # Compute baseline performance
    baseline_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            baseline_loss += loss.item() * x.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    baseline_loss /= len(val_loader.dataset)
    print(f"Baseline MSE Loss: {baseline_loss:.4f}")
    
    # Initialize importance dictionary
    feature_importances = {}
    
    # Convert validation data to a single tensor for manipulation
    all_x = []
    all_y = []
    for x, y in val_loader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x, dim=0).to(device)
    all_y = torch.cat(all_y, dim=0).to(device)
    
    for i, feature in enumerate(feature_columns):
        # Shuffle the i-th feature
        x_permuted = deepcopy(all_x)
        # Shuffle along the batch dimension
        perm = torch.randperm(x_permuted.size(0))
        x_permuted[:, :, i] = x_permuted[perm, :, i]
        
        # Compute loss with permuted feature
        preds = model(x_permuted)
        loss = loss_fn(preds, all_y).item()
        
        # Importance is the increase in loss
        importance = loss - baseline_loss
        feature_importances[feature] = importance
        print(f"Feature: {feature}, Permutation Importance: {importance:.4f}")
    
    return feature_importances