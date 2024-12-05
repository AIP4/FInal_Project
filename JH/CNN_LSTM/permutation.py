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

def compute_cnn_lstm_permutation_importance(model, val_loader, feature_columns, pm_columns, device, loss_fn=nn.L1Loss()):
    model.eval()
    
    # Compute baseline performance
    baseline_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for w_x, pm_x, y in val_loader:
            w_x, pm_x, y = w_x.to(device), pm_x.to(device), y.to(device)
            preds = model(w_x, pm_x)
            loss = loss_fn(preds, y)
            baseline_loss += loss.item() * w_x.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    baseline_loss /= len(val_loader.dataset)
    print(f"Baseline MSE Loss: {baseline_loss:.4f}")

    # Initialize importance dictionary
    feature_importances = {}
    
    # Convert validation data to a single tensor for manipulation
    all_w_x = []
    all_pm_x = []
    all_y = []
    for w_x, pm_x, y in val_loader:
        all_w_x.append(w_x)
        all_pm_x.append(pm_x)
        all_y.append(y)
    all_w_x = torch.cat(all_w_x, dim=0).to(device)
    all_pm_x = torch.cat(all_pm_x, dim=0).to(device)
    all_y = torch.cat(all_y, dim=0).to(device)

    # Feature importance

    # Permute feature columns
    for i, feature in enumerate(feature_columns):
        # Shuffle the i-th feature
        w_x_permuted = deepcopy(all_w_x)
        # Shuffle along the batch dimension
        perm = torch.randperm(w_x_permuted.size(0))
        w_x_permuted[:, :, i] = w_x_permuted[perm, :, i]

        # Compute loss with permuted feature
        preds = model(w_x_permuted, all_pm_x)
        loss = loss_fn(preds, all_y).item()
        
        # Importance is the increase in loss
        importance = loss - baseline_loss
        feature_importances[feature] = importance
        print(f"Feature: {feature}, Permutation Importance: {importance:.4f}")

    # Permute PM columns
    for i, feature in enumerate(pm_columns):
        # Shuffle the i-th feature
        pm_x_permuted = deepcopy(all_pm_x)
        # Shuffle along the batch dimension
        perm = torch.randperm(pm_x_permuted.size(0))
        pm_x_permuted[:, :, i] = pm_x_permuted[perm, :, i]

        # Compute loss with permuted feature
        preds = model(all_w_x, pm_x_permuted)
        loss = loss_fn(preds, all_y).item()
        
        # Importance is the increase in loss
        importance = loss - baseline_loss
        feature_importances[feature] = importance
        print(f"Feature: {feature}, Permutation Importance: {importance:.4f}")
    
    return feature_importances