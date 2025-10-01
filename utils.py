import os
import matplotlib.pyplot as plt
import random
from torch.utils.data import ConcatDataset, Subset


from dataset import CanyonDataset

# --------------------------------------------------------------------------------- #
#                            Some Utility Functions
# --------------------------------------------------------------------------------- #




def LossPlot(all_losses, results_folder, title, filename):
    colors = ['b', 'g', 'r', 'k', 'c', 'm']
    plt.figure(figsize=(8,6))
    for i, (num_imgs, (train_losses, val_losses)) in enumerate(all_losses.items()):
        color = colors[i % len(colors)]
        plt.plot(train_losses, label=f"Train {num_imgs} imgs", linewidth=2, color=color)
        plt.plot(val_losses, '--', label=f"Val {num_imgs} imgs", linewidth=2, color=color)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.savefig(os.path.join(results_folder, filename))
    plt.close()


def print_title(title):
    length = 100
    dashes = "-" * length
    spaces = " " * ((length - len(title)) // 2)
    print(f"\n{dashes}")
    print(f"\n{spaces}{title}")
    print(f"\n{dashes}")