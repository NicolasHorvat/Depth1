import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def test_model(model, test_loader, device, num_samples=5):
    """
    Test the depth prediction model on a given dataset and plot results,
    including an error map (absolute difference between prediction and ground truth).
    """
    model.eval()
    with torch.no_grad():
        for i, (rgb, depth) in enumerate(test_loader):
            rgb = rgb.to(device)
            depth = depth.to(device)
            pred = model(rgb)

            # Select first item in batch
            rgb_np = rgb[0].permute(1,2,0).cpu().numpy()   # [C,H,W] -> [H,W,C]
            depth_np = depth[0].squeeze().cpu().numpy()    # [1,H,W] -> [H,W]
            pred_np = pred[0].squeeze().cpu().numpy()      # [1,H,W] -> [H,W]

            # Compute absolute error
            error_np = abs(depth_np - pred_np)

            # Plot RGB, GT, Prediction, Error
            fig, ax = plt.subplots(1, 4, figsize=(18,4))
            ax[0].imshow(rgb_np)
            ax[0].set_title("RGB Image")
            ax[0].axis('off')

            im1 = ax[1].imshow(depth_np, cmap="viridis")
            ax[1].set_title("Ground Truth Depth [m]")
            fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

            im2 = ax[2].imshow(pred_np, cmap="viridis")
            ax[2].set_title("Predicted Depth [m]")
            fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            im3 = ax[3].imshow(error_np, cmap="magma")
            ax[3].set_title("Absolute Error [m]")
            fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(f"test_{i}.png")
            plt.close(fig)

            if i+1 >= num_samples:
                break

    # Optional: compute average MSE
    mse_total = 0
    count = 0
    with torch.no_grad():
        for rgb, depth in test_loader:
            rgb, depth = rgb.to(device), depth.to(device)
            pred = model(rgb)
            mse_total += ((pred - depth)**2).mean().item()
            count += 1
    print("Average MSE on test set:", mse_total / count)
