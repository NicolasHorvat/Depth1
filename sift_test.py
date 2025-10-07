
import os
import cv2
import torch
import matplotlib.pyplot as plt
from dataset import CanyonDataset



class CanyonDatasetSiftPriors(CanyonDataset):
    def __getitem__(self, idx):
        rgb, depth = super().__getitem__(idx)  # [3,H,W], [1,H,W]
        _, H, W = depth.shape

        # Convert RGB to numpy HxWx3 for OpenCV
        rgb_np = (rgb.permute(1,2,0).numpy() * 255).astype('uint8')
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

        # 16-quadrants
        rows = 4
        cols = 4
        n_keypoints_per_patch = 10
        keypoints_all = []

        row_height = H // rows
        col_width = W // cols

        sift = cv2.SIFT_create()

        for i in range(rows):
            for j in range(cols):
                y0 = i * row_height
                y1 = H if i == rows-1 else (i+1) * row_height
                x0 = j * col_width
                x1 = W if j == cols-1 else (j+1) * col_width

                patch_gray = gray[y0:y1, x0:x1]
                kps = sift.detect(patch_gray, None)
                kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:n_keypoints_per_patch]

                for kp in kps:
                    # Adjust coordinates to full image
                    kp_x = int(kp.pt[0]) + x0
                    kp_y = int(kp.pt[1]) + y0
                    keypoints_all.append((kp_y, kp_x))

        if len(keypoints_all) == 0:
            # fallback: center pixel
            keypoints_all = [(H//2, W//2)]

        ys = torch.tensor([kp[0] for kp in keypoints_all])
        xs = torch.tensor([kp[1] for kp in keypoints_all])

        known_points = torch.stack([ys, xs], dim=1)
        known_depths = depth[0, ys, xs]

        # Initialize prior
        prior = torch.zeros_like(depth)

        # Grid of coordinates
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        yy = yy.unsqueeze(0)  # [1,H,W]
        xx = xx.unsqueeze(0)  # [1,H,W]

        # Compute squared distances to keypoints
        dists = (yy - ys[:, None, None])**2 + (xx - xs[:, None, None])**2  # [num_points,H,W]

        nearest_idx = torch.argmin(dists, dim=0)
        prior[0] = known_depths[nearest_idx]

        # Concatenate prior to RGB
        rgb_with_prior = torch.cat([rgb, prior], dim=0)  # [4,H,W]

        return rgb_with_prior, depth, known_points



def test_sift(rgb_path, depth_path, idx=0):
    # Load dataset
    dataset = CanyonDataset(rgb_path, depth_path)
    rgb, depth = dataset[idx]  # [3,H,W], [1,H,W]

    # Convert to HxWx3 numpy image
    rgb_np = (rgb.permute(1,2,0).numpy() * 255).astype('uint8')
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape

    sift = cv2.SIFT_create()

    n_keypoints = 160
    n_keypoints_per_quadrant = 10


    # ------------------ Full-image SIFT keypoints ---------------

    kps_full = sift.detect(gray, None)
    kps_full = sorted(kps_full, key=lambda kp: kp.response, reverse=True)[:n_keypoints]

    img_full = cv2.drawKeypoints(
        rgb_np, 
        kps_full, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    for kp in kps_full:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img_full, (x, y), radius=5, color=(0,0,0), thickness=-1)


    # ----------- 16-quadrants (4x4) SIFT keypoints -----------------

    rows = 4
    cols = 4
    keypoints_patch = []

    row_height = H // rows
    col_width = W // cols

    for i in range(rows):
        for j in range(cols):
            y0 = i * row_height
            y1 = H if i == rows-1 else (i+1) * row_height
            x0 = j * col_width
            x1 = W if j == cols-1 else (j+1) * col_width

            patch_gray = gray[y0:y1, x0:x1]
            kps = sift.detect(patch_gray, None)
            kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:n_keypoints_per_quadrant]

            for kp in kps:
                kp.pt = (kp.pt[0] + x0, kp.pt[1] + y0)
            keypoints_patch.extend(kps)

    if len(keypoints_patch) == 0:
        keypoints_patch.append(cv2.KeyPoint(x=W//2, y=H//2, _size=10))

    img_patch = cv2.drawKeypoints(
        rgb_np, 
        keypoints_patch, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    for kp in keypoints_patch:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img_patch, (x, y), radius=5, color=(0,0,0), thickness=-1)

    # Draw grid lines
    for i in range(1, rows):
        y = i * row_height
        cv2.line(img_patch, (0, y), (W, y), color=(255,255,0), thickness=2)
    for j in range(1, cols):
        x = j * col_width
        cv2.line(img_patch, (x, 0), (x, H), color=(255,255,0), thickness=2)


    #  ------------- Using the Class -------------------

    dataset_sift = CanyonDatasetSiftPriors(rgb_path, depth_path)
    rgb_with_prior, depth_class, known_points = dataset_sift[idx]

    rgb_class_np = (rgb_with_prior[:3].permute(1,2,0).numpy() * 255).astype('uint8')
    img_class = rgb_class_np.copy()
    for y, x in known_points:
        cv2.circle(img_class, (int(x), int(y)), radius=5, color=(0,0,0), thickness=-1)

    # Prior visualization
    prior = rgb_with_prior[3]  # the 4th channel is the SIFT-based prior
    prior_np = prior.numpy()   # [H, W]


    #  -------------- Plots -----------------
    
    fig, ax = plt.subplots(1, 4, figsize=(32, 8))

    ax[0].imshow(img_full)
    ax[0].set_title(f"Full-image SIFT (Total: {len(kps_full)})")
    ax[0].axis('off')

    ax[1].imshow(img_patch)
    ax[1].set_title(f"16-quadrants SIFT (Total: {len(keypoints_patch)})")
    ax[1].axis('off')

    ax[2].imshow(img_class)
    ax[2].set_title(f"Class-based 16-quadrants SIFT (Total: {len(known_points)})")
    ax[2].axis('off')

    im = ax[3].imshow(prior_np, cmap='cividis')
    ax[3].set_title("SIFT Depth Prior")
    ax[3].axis('off')
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)

    # Save figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "sift_test.png")
    plt.savefig(save_path)
    print(f"Figure saved to: {save_path}")




if __name__ == "__main__":

    canyons_path = r"H:\canyons"  # change to your dataset folder
    flatiron_path = os.path.join(canyons_path, "flatiron")
    flatiron_depth_path = os.path.join(flatiron_path, "depth")
    flatiron_imgs_path = os.path.join(flatiron_path, "imgs")
    flatiron_seaErra_path = os.path.join(flatiron_path, "seaErra")

    test_sift(flatiron_seaErra_path, flatiron_depth_path , idx=0)
