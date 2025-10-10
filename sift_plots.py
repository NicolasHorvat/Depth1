import os
import cv2
import numpy as np
import pickle
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
 
from paths import *
from utils import print_title
from sift import *



def plot_sift_features_matches(imgs_path, seaErra_path, idx=0):
    """
    Visualizes SIFT matches between two frames.
    Compares imgs to seaErra.
    Tests get_and_match_sift_features() function
    """

    # Sort files
    sorted_imgs_path = sorted(os.listdir(imgs_path))
    sorted_seaErra = sorted(os.listdir(seaErra_path))

    # Pick consecutive images
    img1_img_path = os.path.join(imgs_path, sorted_imgs_path[idx])
    img2_img_path = os.path.join(imgs_path, sorted_imgs_path[idx+1])
    img1_seaErra_path = os.path.join(seaErra_path, sorted_seaErra[idx])
    img2_seaErra_path = os.path.join(seaErra_path, sorted_seaErra[idx+1])

    # Read images
    img1_img = cv2.imread(img1_img_path)
    img2_img = cv2.imread(img2_img_path)
    img1_seaErra = cv2.imread(img1_seaErra_path)
    img2_seaErra = cv2.imread(img2_seaErra_path)

    # Compute SIFT matches
    pts1_img, pts2_img = get_and_match_sift_features(img1_img_path, img2_img_path)
    pts1_seaErra, pts2_seaErra = get_and_match_sift_features(img1_seaErra_path, img2_seaErra_path)

    # --- Visualization function ---
    def draw_matches(img1, img2, pts1, pts2):
        h, w = img1.shape[:2]
        vis = np.zeros((h, 2*w, 3), dtype=np.uint8)
        vis[:h, :w] = img1
        vis[:h, w:2*w] = img2

        for i in range(len(pts1)):
            x1, y1 = pts1[i]
            x2, y2 = pts2[i]
            p1 = (int(x1), int(y1))
            p2 = (int(x2) + w, int(y2))
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(vis, p1, 4, color, -1)
            cv2.circle(vis, p2, 4, color, -1)
            cv2.line(vis, p1, p2, color, 1)
        return vis

    vis_img = draw_matches(img1_img, img2_img, pts1_img, pts2_img)
    vis_seaErra = draw_matches(img1_seaErra, img2_seaErra, pts1_seaErra, pts2_seaErra)

    # --- Plot ---
    fig, axs = plt.subplots(2, 1, figsize=(15, 12), facecolor='lightgray')
    title_fontsize = 22

    axs[0].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"{len(pts1_img)} matched SIFT features (imgs)", fontsize=title_fontsize)
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(vis_seaErra, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"{len(pts1_seaErra)} matched SIFT features (seaErra)", fontsize=title_fontsize)
    axs[1].axis('off')

    plt.tight_layout()

    # save
    canyon = os.path.basename(os.path.dirname(os.path.normpath(imgs_path)))
    filename = f"sift_features_matches_plot_{canyon}_idx{idx}.png"

    save_path = os.path.join(plots_folder, filename)
    plt.savefig(save_path)
    print(f"Saved SIFT features matches plot to {save_path}")


def plot_sift_priors(imgs_path, seaErra_path, depth_path, idx=500):
    """
    Visualizes Sift Priors
    """

    # ----------- img seaErra depth -----------
    img_file = os.path.join(imgs_path, sorted(os.listdir(imgs_path))[idx])
    sea_file = os.path.join(seaErra_path, sorted(os.listdir(seaErra_path))[idx])
    depth_file = os.path.join(depth_path, sorted(os.listdir(depth_path))[idx])

    img = cv2.imread(img_file)
    seaErra = cv2.imread(sea_file)
    depth_map = tifffile.imread(depth_file).astype(np.float32)

    # ----------- Get sift features -----------
    _, pts2_imgs = get_saved_sift_features_points(imgs_path, idx)
    xs_imgs = np.clip(pts2_imgs[:, 0].astype(int), 0, depth_map.shape[1]-1)
    ys_imgs = np.clip(pts2_imgs[:, 1].astype(int), 0, depth_map.shape[0]-1)

    _, pts2_seaErra = get_saved_sift_features_points(seaErra_path, idx)
    xs_seaErra = np.clip(pts2_seaErra[:, 0].astype(int), 0, depth_map.shape[1]-1)
    ys_seaErra = np.clip(pts2_seaErra[:, 1].astype(int), 0, depth_map.shape[0]-1)

    # ----- get Nearest-neighbor prior -----
    nn_prior_imgs = get_nearest_neighbor_prior(imgs_path, depth_path, idx)
    nn_prior_seaErra = get_nearest_neighbor_prior(seaErra_path, depth_path, idx)

    # ----- get gaussian prior -----
    gauss_prior_imgs = get_gaussian_prior(imgs_path, idx, sigma = 20)
    gauss_prior_seaErra = get_gaussian_prior(seaErra_path, idx, sigma = 20)


    # ----- Plots -----
    fig, axs = plt.subplots(2, 4, figsize=(20, 12), facecolor='lightgray')
    colormap = 'cividis'
    points_size = 3
    title_fontsize = 22
    depth_min = 0.0
    depth_max = depth_map.max()

    # Row 1: imgs
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].scatter(xs_imgs, ys_imgs, c='yellow', s=points_size)
    axs[0, 0].set_title("RGB Image (imgs)", fontsize=title_fontsize)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(nn_prior_imgs, cmap=colormap, vmin=depth_min, vmax=depth_max)
    axs[0, 1].scatter(xs_imgs, ys_imgs, c='k', s=points_size)
    axs[0, 1].set_title("nn Prior (imgs)", fontsize=title_fontsize)
    axs[0, 1].axis('off')

    axs[0, 2].imshow(gauss_prior_imgs, cmap=colormap)
    axs[0, 2].scatter(xs_imgs, ys_imgs, c='k', s=points_size)
    axs[0, 2].set_title("Gaussian Prior (imgs)", fontsize=title_fontsize)
    axs[0, 2].axis('off')

    axs[0, 3].imshow(depth_map, cmap=colormap, vmin=depth_min, vmax=depth_max)
    axs[0, 3].scatter(xs_imgs, ys_imgs, c='k', s=points_size)
    axs[0, 3].set_title("Ground Truth Depth Map", fontsize=title_fontsize)
    axs[0, 3].axis('off')

    # Row 2: SeaErra
    axs[1, 0].imshow(cv2.cvtColor(seaErra, cv2.COLOR_BGR2RGB))
    axs[1, 0].scatter(xs_seaErra, ys_seaErra, c='yellow', s=points_size)
    axs[1, 0].set_title("RGB Image (SeaErra)", fontsize=title_fontsize)
    axs[1, 0].axis('off')

    axs[1, 1].imshow(nn_prior_seaErra, cmap=colormap, vmin=depth_min, vmax=depth_max)
    axs[1, 1].scatter(xs_seaErra, ys_seaErra, c='k', s=points_size)
    axs[1, 1].set_title("nn Prior (SeaErra)", fontsize=title_fontsize)
    axs[1, 1].axis('off')

    axs[1, 2].imshow(gauss_prior_seaErra, cmap=colormap)
    axs[1, 2].scatter(xs_seaErra, ys_seaErra, c='k', s=points_size)
    axs[1, 2].set_title("Gaussian Prior (SeaErra)", fontsize=title_fontsize)
    axs[1, 2].axis('off')

    axs[1, 3].imshow(depth_map, cmap=colormap, vmin=depth_min, vmax=depth_max)
    axs[1, 3].scatter(xs_seaErra, ys_seaErra, c='k', s=points_size)
    axs[1, 3].set_title("Ground Truth Depth Map", fontsize=title_fontsize)
    axs[1, 3].axis('off')

    plt.tight_layout()
    
    #save
    canyon = os.path.basename(os.path.dirname(os.path.normpath(imgs_path)))
    filename = f"sift_priors_plot_{canyon}_idx{idx}.png"

    save_path = os.path.join(plots_folder, filename)
    plt.savefig(save_path)
    print(f"Saved SIFT prior plot to {save_path}")


# ----------------------------------------------------------------------------
#                       Main (sift_plots.py)
# ----------------------------------------------------------------------------
if __name__ == "__main__":

    print_title("sift_plots.py")

    # Folder to save plots
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_folder = os.path.join(script_dir, "sift_plots")
    os.makedirs(plots_folder, exist_ok=True)


    indices_to_plot = [100, 800]

    for imgs_path, seaErra_path, depth_path in canyon_imgs_seaErra_depth_paths:
        for idx in indices_to_plot:
            print(f"\nPlotting SIFT prior for canyon: {imgs_path}, frame idx: {idx}")
            plot_sift_priors(imgs_path, seaErra_path, depth_path, idx=idx) 


    for imgs_path, seaErra_path, _ in canyon_imgs_seaErra_depth_paths:
        for idx in indices_to_plot:
            print(f"\nPlotting SIFT matches for canyon: {imgs_path}, frame idx: {idx}")
            plot_sift_features_matches(imgs_path, seaErra_path, idx=idx)

    # way more sift features detected in seaErra set than in imgs set

