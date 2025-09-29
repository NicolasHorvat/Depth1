import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

def test_model(model, test_loader, device, results_folder, num_samples=5, name = ''):
    """
    Test the model and plot examples
    """


    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            rgb, depth = batch
            rgb, depth = rgb.to(device), depth.to(device)
            pred = model(rgb)

            # Select first item in batch
            rgb_np = rgb[0].permute(1,2,0).cpu().numpy()   # [C,H,W] -> [H,W,C]
            depth_np = depth[0].squeeze().cpu().numpy()    # [1,H,W] -> [H,W]
            pred_np = pred[0].squeeze().cpu().numpy()      # [1,H,W] -> [H,W]

            # Compute absolute error
            error_np = abs(depth_np - pred_np)

            depth_vmin, depth_vmax = 0, depth_np.max()
            pred_vmin, pred_vmax = 0, pred_np.max()
            error_vmin, error_vmax = 0, error_np.max()

            # Plot RGB, GT, Prediction, Error
            fig, ax = plt.subplots(1, 4, figsize=(18,4))
            ax[0].imshow(rgb_np)
            ax[0].set_title("RGB Image")
            ax[0].axis('off')

            im1 = ax[1].imshow(depth_np, cmap="viridis", vmin=depth_vmin, vmax=depth_vmax)
            ax[1].set_title("Ground Truth Depth [m]")
            fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

            im2 = ax[2].imshow(pred_np, cmap="viridis", vmin=pred_vmin, vmax=pred_vmax)
            ax[2].set_title("Predicted Depth [m]")
            fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            im3 = ax[3].imshow(error_np, cmap="viridis", vmin=error_vmin, vmax=error_vmax)
            ax[3].set_title("Absolute Error [m]")
            fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, f"test_{name}_{i}.png"))
            plt.close(fig)

            if i+1 >= num_samples:
                break


    criterion = nn.MSELoss()
    mse_total = 0.0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            rgb, depth = batch
            rgb, depth = rgb.to(device), depth.to(device)
            pred = model(rgb)

            mse_total += criterion(pred, depth).item()
            count += 1

    avg_mse = mse_total / count
    print("Average MSE on test set:", avg_mse)
