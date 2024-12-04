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

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train(train_dataset,
          valid_dataset,
          feature_columns,
          region,
          model_name,
          config,
          device):
    os.makedirs(f"models/{model_name}", exist_ok=True)
    os.makedirs(f"logs/{model_name}", exist_ok=True)

    set_logger(f"logs/{model_name}/{region}_ts={config['output_size']}.log")

    # Model
    model = FinedustLSTM(input_size=len(feature_columns),
                         hidden_size=config['hidden_size'],
                         num_layers=config['num_layers'],
                         output_size=config['output_size'],
                         dropout_prob=config['dropout']).to(device)

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

    total_preds = []
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
            train_loss += loss.item()

        
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        epoch_preds = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item()
                epoch_preds.append(pred.cpu().numpy())
        
        predictions_array = np.concatenate(epoch_preds)
        
        val_loss /= len(val_loader)

        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"models/{model_name}/{region}.pth")
            logging.info(f"Model saved with loss: {best_loss:.4f}")
            epochs_without_improvement = 0
            total_preds.append(predictions_array)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config['patience']:
                logging.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                break
            losses.append((train_loss, val_loss))

    total_preds = np.stack(total_preds, axis=0)
    losses_array = np.array(losses)

    logging.info(f"Training completed with best loss: {best_loss:.4f}")
    return total_preds[-1], losses_array


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