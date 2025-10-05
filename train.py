import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import time
from datetime import datetime

from utils import save_model_at

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


def train_model(train_loader, val_loader, model, num_epochs, results_folder_path, model_name):
    """
    Trains the model.
    Works for DepthNet and DepthNetWithPrior.
    Handles invalid depth pixels (0 or NaN in the ground truth depth) by masking.

    Args:
            train_loader, val_loader:   Dataloaders
            model:                      Model to be trained
            num_epochs:                 number of epochs to be trained
            model_name:                 for naming folders and such
    Returns:
            model:          trained model
            device:         cpu or gpu/cuda
            train_losses:   training losses
            val_losses:     validation losses
            
    """

    start_time = time.time()
    
    # Hyperparameters
    lr = 1e-3
    step_size = 1
    gamma = 0.9


    # Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # optimizer
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    criterion_mse = nn.MSELoss(reduction='none')  # elementwise
    criterion_log = LogLoss()

    # Print Statements
    print("\n")
    print("\\"*50)
    print("Starting training!")
    print(f"Model: {type(model).__name__}")
    print(f"Device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {lr}, Step size: {step_size}, Gamma: {gamma}")
    print(f"Time: {datetime.now().strftime('%H:%M')}")
    print("\\"*50)


    # Losses
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = -1


    for epoch in range(num_epochs): 
        
        epoch_start = time.time()
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
        running_loss = 0.0

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

                running_loss += loss.item() * depth.size(0)

        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step()

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            # Save model
            save_model_at(model, results_folder_path, model_name = model_name)

        # Print some Information
        print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        elapsed_time = epoch_end - start_time
        remaining_time = epoch_duration * (num_epochs - epoch - 1)
        print(f"Epoch {epoch+1} finished in {epoch_duration:.1f}s")
        print(f"Elapsed time: {elapsed_time/60:.1f} min, Estimated remaining time: {remaining_time/60:.1f} min")

    
    print("Finished Training")
    print(f"Best model saved from epoch {best_epoch+1} with Val Loss: {best_val_loss:.6f}")

    return model, device, train_losses, val_losses