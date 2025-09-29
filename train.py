import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import DepthNet, DepthNetWithPrior

class LogLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, mask=None):
        pred = torch.clamp(pred, min=self.eps)
        target = torch.clamp(target, min=self.eps)

        log_diff = torch.log(pred) - torch.log(target)

        if mask is not None:
            log_diff = log_diff[mask]

        return (log_diff ** 2).mean()


def train_model(train_loader, val_loader, results_folder, model, num_epochs, name=None):
    """
    Trains the model.
    Works for DepthNet and DepthNetWithPrior.
    Handles invalid depth pixels by masking.
    """

    # Hyperparameters
    lr = 1e-3
    step_size = 1
    gamma = 0.9


    # Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # optimiser
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    criterion_mse = nn.MSELoss(reduction='none')  # elementwise
    criterion_log = LogLoss()

    train_losses = []
    val_losses = []


    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs} - Training...")

        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc="Training batches"):

            rgb, depth = batch
            rgb, depth = rgb.to(device), depth.to(device)

            optimizer.zero_grad()
            pred = model(rgb)

            # Mask invalid depth pixels (0 and NaN)
            valid_mask = (depth > 0) & (~torch.isnan(depth))

            # Compute losses
            mse_map = criterion_mse(pred, depth)
            mse_loss = mse_map[valid_mask].mean()
            log_loss = criterion_log(pred, depth, mask=valid_mask)
            loss = 0.7 * mse_loss + 0.3 * log_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * depth.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)



        print(f"Epoch {epoch+1}/{num_epochs} - Validation...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation batches"):
                rgb, depth = batch
                rgb, depth = rgb.to(device), depth.to(device)
                pred = model(rgb)

                # Mask invalid depth pixels (0 and NaN)
                valid_mask = (depth > 0) & (~torch.isnan(depth))

                # Compute losses
                mse_map = criterion_mse(pred, depth)
                mse_loss = mse_map[valid_mask].mean()

                log_loss = criterion_log(pred, depth, mask=valid_mask)

                loss = 0.7 * mse_loss + 0.3 * log_loss
                val_loss += loss.item() * depth.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step()
        print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

    
    # Save model
    if name is not None:
        model_path = os.path.join(results_folder, f"depth_model_{name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved as {model_path}")

    return model, device, train_losses, val_losses
