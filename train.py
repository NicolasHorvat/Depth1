import torch
import torch.optim as optim
from tqdm import tqdm
import time
from datetime import datetime
import copy

from utils import save_model_at, combined_loss


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
    step_size = 2
    gamma = 0.9


    # Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # optimizer
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    criterion = combined_loss

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

    best_model_state = None

    #for epoch in range(num_epochs):
    for epoch in tqdm(range(num_epochs), desc="Progress", ncols = 100, leave= True):
        
        epoch_start = time.time()

        model.train()
        running_loss = 0.0

       #for batch in tqdm(train_loader, desc="Training batches", ncols = 100, leave = False):
        for batch in train_loader:

            rgb, depth = batch[:2] # only first two returns
            rgb, depth = rgb.to(device), depth.to(device)

            optimizer.zero_grad()
            pred = model(rgb)

            # Mask invalid depth pixels (0 and NaN)
            valid_mask = (depth > 0) & (~torch.isnan(depth))

            # Compute loss
            batch_loss = criterion(pred, depth, valid_mask)

            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)


        model.eval()
        running_loss = 0.0

        with torch.no_grad():
           #for batch in tqdm(val_loader, desc="Validation batches", ncols = 100, leave = False):
            for batch in val_loader:
                rgb, depth = batch[:2]
                rgb, depth = rgb.to(device), depth.to(device)
                pred = model(rgb)

                # Mask invalid depth pixels (0 and NaN)
                valid_mask = (depth > 0) & (~torch.isnan(depth))

                # Compute losses
                batch_loss = criterion(pred, depth, valid_mask)
                running_loss += batch_loss.item()

        epoch_val_loss = running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        scheduler.step()

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            # Save model
            print("\n")
            save_model_at(model, results_folder_path, model_name = model_name)

        # Print some Information
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        elapsed_time = epoch_end - start_time
        remaining_time = epoch_duration * (num_epochs - epoch - 1)
        print(f"--> Epoch {epoch+1} - (Train Loss: {epoch_train_loss:.4f}), (Val Loss: {epoch_val_loss:.4f}), "
              f"Epoch {epoch+1} finished in {epoch_duration:.2f}s, "
              f"(Elapsed time: {elapsed_time/60:.2f} min), (Estimated remaining time: {remaining_time/60:.2f} min)")

    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    end_time = time.time()
    train_time = end_time - start_time

    print("\n")
    print("\\"*50)
    print(f"- Training Finished!")
    print(f"- Best model saved from epoch {best_epoch+1} with Val Loss: {best_val_loss:.6f}")
    print(f"- Training time: {train_time/60:.1f} min ({train_time/3600:.1f} h)")
    print("\\"*50)

    return model, device, train_losses, val_losses, train_time