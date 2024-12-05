from tqdm import tqdm
from utils import set_logger
from dataset import FinedustDataset
from model import FinedustLSTM
from utils import prepare_data
from sklearn.model_selection import train_test_split
import os
import logging
from interpolate import simple_interpolate
import numpy as np
import math

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train(model,
          train_dataset,
          valid_dataset,
          feature_columns,
          region,
          model_name,
          config,
          device,
          log_suffix=None):
    os.makedirs(f"models/{model_name}", exist_ok=True)
    os.makedirs(f"logs/{model_name}", exist_ok=True)

    if log_suffix is not None:
        set_logger(f"logs/{model_name}/{region}_ts={config['output_size']}.log")
    else:
        set_logger(f"logs/{model_name}/{region}_ts={config['output_size']}_{log_suffix}.log")

    # Model
    logging.info(f"Training model with {region}")
    logging.info(f"Config: {config}")
    logging.info(f"Model: {model}\n")

    logging.info(f"Train dataset: {len(train_dataset)}")
    logging.info(f"Valid dataset: {len(valid_dataset)}\n")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)

    # Optimizer and Loss Function
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.MSELoss()   # MSE Loss

    epochs = config['epochs']
    best_loss = float('inf')
    epochs_without_improvement = 0  # Early stopping

    results_array = []
    losses = []

    # Training
    for epoch in tqdm(range(epochs)):
        # Train
        
        model.train()
        train_loss = 0

        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += math.sqrt(loss.item())

        
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        epoch_results = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += math.sqrt(loss.item())
                epoch_results.append(np.concatenate((pred.cpu().numpy(), y.cpu().numpy()), axis=1))
        
        epoch_results = np.concatenate(epoch_results)
        
        val_loss /= len(val_loader)

        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"models/{model_name}/{region}.pth")
            logging.info(f"Model saved with loss: {best_loss:.4f}")
            epochs_without_improvement = 0
            results_array.append(epoch_results)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config['patience']:
                logging.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                break
        losses.append((train_loss, val_loss))

    results_array = np.stack(results_array, axis=0)
    losses_array = np.array(losses)

    logging.info(f"Training completed with best loss: {best_loss:.4f}")
    
    if log_suffix is not None:
        region = f"{region}_{log_suffix}"

    save_loss_figure(losses_array, model_name, region)
    save_results_figure(results_array, model_name, region, losses_array)
    
    return results_array[-1], losses_array


def validate(model, val_loader, loss_fn, device):
    model.eval()
    preds = []
    val_loss = 0
    for batch in val_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        val_loss += loss.item()
        preds.append(pred.cpu().numpy())
    val_loss /= len(val_loader)
    return val_loss, preds
    

import matplotlib.pyplot as plt

def save_loss_figure(losses_array, model_name, region):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(losses_array) + 1)
    train_losses = losses_array[:, 0]
    val_losses = losses_array[:, 1]

    plt.plot(epochs, train_losses, label="Train", color="blue")
    plt.plot(epochs, val_losses, label="Validation", color="orange")
    plt.title(f"{model_name} Loss (Region={region})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    output_path = f"logs/{model_name}/{region}_loss_curve.png"
    plt.savefig(output_path)
    plt.close()  # Close the plot to free memory
    

def save_results_figure(results_array, model_name, region, loss_list):
    plt.figure(figsize=(18, 6))
    pred_y = results_array[0]
    true_y = results_array[1]
    
    plt.plot(true_y[:1000], label="True", color="blue")
    plt.plot(pred_y[:1000], label="Prediction", color="orange", linestyle="dashed")
    plt.title(f"LSTM (Region={region}, loss={np.min(loss_list[:, 1]): .2f})")
    plt.legend()

    output_path = f"logs/{model_name}/{region}_results_curve.png"
    plt.savefig(output_path)
    plt.close()

