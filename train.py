import torch
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
import torch.nn as nn
from model import SimpleDepthNet
from dataset import CanyonDataset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train_model(dataset, num_epochs, batch_size):
    """
    Train the SimpleDepthNet model on a given dataset.

    Args:
        dataset: CanyonDataset instance
        num_epochs: number of epochs to train
        batch_size: batch size

    Returns:
        model: trained model
        val_loader: validation DataLoader
        device: device used (CPU/GPU)
    """
    # Split dataset into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model, optimizer, loss
    model = SimpleDepthNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} - Training...")
        model.train()
        running_loss = 0.0
        for rgb, depth in tqdm(train_loader, desc="Training batches"):
            rgb, depth = rgb.to(device), depth.to(device)
            optimizer.zero_grad()
            pred = model(rgb)
            loss = criterion(pred, depth)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * rgb.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        print(f"Epoch {epoch+1}/{num_epochs} - Validation...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb, depth in tqdm(val_loader, desc="Validation batches"):
                rgb, depth = rgb.to(device), depth.to(device)
                pred = model(rgb)
                val_loss += criterion(pred, depth).item() * rgb.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

    # Plot losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    print("Loss plot saved as loss_plot.png")

    # Save model
    torch.save(model.state_dict(), "simple_depth_model.pth")
    print("Model saved as simple_depth_model.pth")

    return model, val_loader, device
