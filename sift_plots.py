import os
import cv2
import numpy as np
import pickle
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
 
from paths import *
from utils import print_title
from sift import match_sift_features, get_canyon_sift_features_path

def visualize_sift_matches(tiny_canyon_imgs_path, tiny_canyon_seaErra_path, idx=0):
    """
    Visualizes SIFT matches between two frames.
    Compares imgs to seaErra.
    """

    # Sort files
    sorted_tiny_canyon_imgs = sorted(os.listdir(tiny_canyon_imgs_path))
    sorted_tiny_canyon_seaErra = sorted(os.listdir(tiny_canyon_seaErra_path))

    # Pick consecutive images
    img1_img_path = os.path.join(tiny_canyon_imgs_path, sorted_tiny_canyon_imgs[idx])
    img2_img_path = os.path.join(tiny_canyon_imgs_path, sorted_tiny_canyon_imgs[idx+1])
    img1_seaErra_path = os.path.join(tiny_canyon_seaErra_path, sorted_tiny_canyon_seaErra[idx])
    img2_seaErra_path = os.path.join(tiny_canyon_seaErra_path, sorted_tiny_canyon_seaErra[idx+1])

    # Read images
    img1_img = cv2.imread(img1_img_path)
    img2_img = cv2.imread(img2_img_path)
    img1_seaErra = cv2.imread(img1_seaErra_path)
    img2_seaErra = cv2.imread(img2_seaErra_path)

    # Compute SIFT matches
    pts1_img, pts2_img = match_sift_features(img1_img, img2_img)
    pts1_seaErra, pts2_seaErra = match_sift_features(img1_seaErra, img2_seaErra)

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
    axs[0].set_title(f"SIFT matches (imgs): {len(pts1_img)} inliers", fontsize=title_fontsize)
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(vis_seaErra, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"SIFT matches (SeaErra): {len(pts1_seaErra)} inliers", fontsize=title_fontsize)
    axs[1].axis('off')

    plt.tight_layout()

    # Save
    filename = f"sift_matches_idx{idx}.png"
    save_path = os.path.join(plots_folder, filename)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved SIFT matches plot to {save_path}")


def visualize_sift_prior(imgs_path, seaErra_path, depth_path, idx=500):
    """
    Visualizes Sift Priors
    """

    # Load SIFT matches
    matches_file_imgs = get_canyon_sift_features_path(imgs_path)
    matches_file_seaErra = get_canyon_sift_features_path(seaErra_path)
    
    with open(matches_file_imgs, 'rb') as f:
        all_matches_imgs = pickle.load(f)
    with open(matches_file_seaErra, 'rb') as f:
        all_matches_seaErra = pickle.load(f)

    sorted_imgs = sorted(os.listdir(imgs_path))
    sorted_sea = sorted(os.listdir(seaErra_path))
    sorted_depth = sorted(os.listdir(depth_path))

    # ----------- imgs -----------
    img_file = os.path.join(imgs_path, sorted_imgs[idx])
    depth_file = os.path.join(depth_path, sorted_depth[idx])
    img_frame = cv2.imread(img_file)
    depth_frame = tifffile.imread(depth_file).astype(np.float32)
    H, W = img_frame.shape[:2]

    key_imgs = f"{sorted_imgs[idx-1]}-{sorted_imgs[idx]}"
    pts2_imgs = all_matches_imgs[key_imgs]['pts2']
    xs_imgs = np.clip(pts2_imgs[:, 0].astype(int), 0, W-1)
    ys_imgs = np.clip(pts2_imgs[:, 1].astype(int), 0, H-1)
    depths_at_pts_imgs = depth_frame[ys_imgs, xs_imgs]

    # Prior for imgs
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dists = (yy.flatten()[:, None] - ys_imgs[None, :])**2 + (xx.flatten()[:, None] - xs_imgs[None, :])**2
    nearest_idx = np.argmin(dists, axis=1)
    prior_imgs = np.zeros_like(depth_frame)
    prior_imgs[:, :] = depths_at_pts_imgs[nearest_idx].reshape(H, W)

    # ----- SeaErra image -----
    sea_file = os.path.join(seaErra_path, sorted_sea[idx])
    sea_frame = cv2.imread(sea_file)
    key_sea = f"{sorted_sea[idx-1]}-{sorted_sea[idx]}"
    pts2_sea = all_matches_seaErra[key_sea]['pts2']
    xs_sea = np.clip(pts2_sea[:, 0].astype(int), 0, W-1)
    ys_sea = np.clip(pts2_sea[:, 1].astype(int), 0, H-1)
    depths_at_pts_sea = depth_frame[ys_sea, xs_sea]

    # Prior for SeaErra
    dists = (yy.flatten()[:, None] - ys_sea[None, :])**2 + (xx.flatten()[:, None] - xs_sea[None, :])**2
    nearest_idx = np.argmin(dists, axis=1)
    prior_sea = np.zeros_like(depth_frame)
    prior_sea[:, :] = depths_at_pts_sea[nearest_idx].reshape(H, W)

    # ----- Plot -----
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), facecolor='lightgray')
    colormap = 'cividis'
    points_size = 10
    title_fontsize = 22

    # Row 1: imgs
    axs[0, 0].imshow(cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
    axs[0, 0].scatter(xs_imgs, ys_imgs, c='yellow', s=points_size)
    axs[0, 0].set_title("RGB Image (imgs)", fontsize = title_fontsize)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(prior_imgs, cmap=colormap)
    axs[0, 1].scatter(xs_imgs, ys_imgs, c='k', s=points_size)
    axs[0, 1].set_title("SIFT Prior (imgs)", fontsize = title_fontsize)
    axs[0, 1].axis('off')

    axs[0, 2].imshow(depth_frame, cmap=colormap)
    axs[0, 2].scatter(xs_imgs, ys_imgs, c='k', s=points_size)
    axs[0, 2].set_title("Ground Truth Depth Map (imgs)", fontsize = title_fontsize)
    axs[0, 2].axis('off')

    # Row 2: SeaErra
    axs[1, 0].imshow(cv2.cvtColor(sea_frame, cv2.COLOR_BGR2RGB))
    axs[1, 0].scatter(xs_sea, ys_sea, c='yellow', s=points_size)
    axs[1, 0].set_title("RGB Image (SeaErra)", fontsize = title_fontsize)
    axs[1, 0].axis('off')

    axs[1, 1].imshow(prior_sea, cmap=colormap)
    axs[1, 1].scatter(xs_sea, ys_sea, c='k', s=points_size)
    axs[1, 1].set_title("SIFT Prior (SeaErra)", fontsize = title_fontsize)
    axs[1, 1].axis('off')

    axs[1, 2].imshow(depth_frame, cmap=colormap)
    axs[1, 2].scatter(xs_sea, ys_sea, c='k', s=points_size)
    axs[1, 2].set_title("Ground Truth Depth Map (SeaErra)", fontsize = title_fontsize)
    axs[1, 2].axis('off')

    plt.tight_layout()
    
    #save
    filename = f"sift_prior_plot.png"
    save_path = os.path.join(plots_folder, filename)
    plt.savefig(save_path)
    print(f"\nSaved SIFT prior plot to {save_path}")


# ----------------------------------------------------------------------------
#                       Main (sift_plots.py)
# ----------------------------------------------------------------------------
if __name__ == "__main__":

    print_title("sift_plots.py")

    # Folder to save plots
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_folder = os.path.join(script_dir, "sift_plots")
    os.makedirs(plots_folder, exist_ok=True)


    visualize_sift_prior(tiny_canyon_imgs_path, tiny_canyon_seaErra_path, tiny_canyon_depth_path)

    visualize_sift_matches(tiny_canyon_imgs_path, tiny_canyon_seaErra_path, idx = 500)

    # way more sift features detected in seaErra set than in imgs set






    # TODO: Save Sift priors channes
