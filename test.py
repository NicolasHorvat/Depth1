import os
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

from utils import combined_loss

def test_model(model, test_loader, device, results_folder, num_samples = 5, model_name = 'noName'):
    """
    Test the model and plot some plots
    """

    print("\\"*50)
    print("Testing Dataset...")

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            rgb, depth = batch[:2]
            rgb, depth = rgb.to(device), depth.to(device)
            pred = model(rgb)

            # Select first item in batch
            rgb_np = rgb[0].permute(1,2,0).cpu().numpy()
            rgb_np = rgb_np[:, :, :3]
            depth_np = depth[0].squeeze().cpu().numpy()
            pred_np = pred[0].squeeze().cpu().numpy()

            # Compute absolute error
            error_np = np.abs(depth_np - pred_np)

            # Mask
            valid_mask = depth_np > 0
            pred_masked = np.where(valid_mask, pred_np, 0)
            error_masked = np.where(valid_mask, error_np, 0)

            # colorbar limits
            depth_pred_vmax = max(depth_np.max(), pred_np.max())
            depth_vmin, depth_vmax = 0, depth_pred_vmax
            error_vmin, error_vmax = 0, error_np.max()


            # Plotting
            fig, ax = plt.subplots(2, 3, figsize=(20, 12), facecolor='lightgray')

            # Compute masked MSE for the first item in batch
            valid_mask = (depth[0] > 0) & (~torch.isnan(depth[0]))
            loss = combined_loss(pred[0], depth[0], mask=valid_mask).item()

            # Add MSE to the figure title
            fig.suptitle(f"{model_name} - Test Sample {i} - Masked Loss: {loss:.4f}", fontsize=24)

            title_fontsize = 22

            # Row 1 (unmasked)
            ax[0,0].imshow(rgb_np)
            ax[0,0].set_title("RGB Image", fontsize = title_fontsize)
            ax[0,0].axis('off')

            colormap = "cividis" # ,, "viridis", ,  "plasma" "inferno" "magma""gray"

            im4 = ax[0,1].imshow(pred_np, cmap = colormap, vmin = depth_vmin, vmax = depth_vmax)
            ax[0,1].set_title("Predicted Depth [m]", fontsize = title_fontsize)
            ax[0,1].axis('off')
            fig.colorbar(im4, ax=ax[0,1], fraction=0.046, pad=0.04)

            im5 = ax[0,2].imshow(error_np, cmap = colormap, vmin = error_vmin, vmax = error_vmax)
            ax[0,2].set_title("Absolute Error [m]", fontsize = title_fontsize)
            ax[0,2].axis('off')
            fig.colorbar(im5, ax=ax[0,2], fraction=0.046, pad=0.04)

            # Row 2 (masked)
            im1 = ax[1,0].imshow(depth_np, cmap = colormap, vmin=depth_vmin, vmax = depth_vmax)
            ax[1,0].set_title("Ground Truth Depth [m]", fontsize = title_fontsize)
            ax[1,0].axis('off')
            fig.colorbar(im1, ax=ax[1,0], fraction=0.046, pad=0.04)

            im2 = ax[1,1].imshow(pred_masked, cmap = colormap, vmin=depth_vmin, vmax = depth_vmax)
            ax[1,1].set_title("Predicted Depth (Masked)", fontsize = title_fontsize)
            ax[1,1].axis('off')
            fig.colorbar(im2, ax=ax[1,1], fraction=0.046, pad=0.04)

            im3 = ax[1,2].imshow(error_masked, cmap = colormap, vmin = error_vmin, vmax = error_vmax)
            ax[1,2].set_title("Absolute Error (Masked)", fontsize = title_fontsize)
            ax[1,2].axis('off')
            fig.colorbar(im3, ax=ax[1,2], fraction=0.046, pad=0.04)

            

            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, f"test_grid_{model_name}_{i}.png"))
            plt.close(fig)

            if i+1 >= num_samples:
                break


    # test loss
    running_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            rgb, depth = batch[:2]
            rgb, depth = rgb.to(device), depth.to(device)
            pred = model(rgb)

            valid_mask = (depth > 0) & (~torch.isnan(depth))
            batch_loss = combined_loss(pred, depth, mask=valid_mask)
            running_loss += batch_loss.item()

    test_loss = running_loss / len(test_loader)
    print(f"Average masked combined loss on test set:, {test_loss:.4f}")
    print("\\"*50)
    
    return test_loss