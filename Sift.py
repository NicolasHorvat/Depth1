import os
import cv2
import numpy as np
import pickle
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
 
from paths import *
from utils import print_title


# --------------- functions for Sift Features  -----------------------

def get_sift_features(img_path):
    """
    Computes SIFT keypoints and descriptors for a single image.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(img, None)

    if des is None:
        des = np.zeros((0, 128), dtype=np.float32)
        kps = []

    return kps, des


def match_sift_features(kps1, des1, kps2, des2):
    """
    Matches SIFT descriptors for two sets of keypoints and descriptors.
    """

    if des1.shape[0] == 0 or des2.shape[0] == 0:
        print("Warning: One of the images has no descriptors.")
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    # Match Descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for mn in matches:
        if len(mn) != 2:
            continue
        m, n = mn
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        print("Warning: Not enough good matches found (<8).")
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None:
        print("Warning: Fundamental matrix could not be computed (F is None).")
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    inliers = mask.ravel() == 1

    if np.sum(inliers) == 0:
        print("Warning: No inliers found after RANSAC filtering.")
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    return pts1[inliers], pts2[inliers]


def get_and_match_sift_features(img1_path, img2_path):
    """
    Detects and matches SIFT features between two frames.
    """
    kps1, des1 = get_sift_features(img1_path)
    kps2, des2 = get_sift_features(img2_path)

    return match_sift_features(kps1, des1, kps2, des2)


def save_all_matched_sift_features_of_dataset(imgs_folder):
    '''
    creates and saves matched sift features for a given dataset
    saves them in the canyons_sift_features folder
    Returns the path
    '''

    # save path
    canyon_name = os.path.basename(os.path.dirname(os.path.normpath(imgs_folder)))
    folder_name = os.path.basename(os.path.normpath(imgs_folder))
    print(f"Computing Sift matches for {canyon_name}_{folder_name}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "canyons_sift_features")
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, f'sift_features_{canyon_name}_{folder_name}.pkl')
    if os.path.exists(save_path):
        print(f"-> Sift features for {canyon_name}_{folder_name} already exist at {save_path} ->  Skipping computation")
        return save_path 

    sorted_imgs = sorted(os.listdir(imgs_folder))

    # keypoints and descriptors for all images
    all_keypoints = []
    all_descriptors = []

    for img in tqdm(sorted_imgs, desc="Computing SIFT per image", ncols = 100, leave= True):
        img_path = os.path.join(imgs_folder, img)
        kps, des = get_sift_features(img_path)
        all_keypoints.append(kps)
        all_descriptors.append(des)

    # Match consecutive image pairs
    all_matches = {}
    for i in tqdm(range(len(sorted_imgs)-1), desc="Matching consecutive images", ncols = 100, leave= True):
        kps1, kps2 = all_keypoints[i], all_keypoints[i+1]
        des1, des2 = all_descriptors[i], all_descriptors[i+1]

        pts1, pts2 = match_sift_features(kps1, des1, kps2, des2)

        all_matches[f"{sorted_imgs[i]}-{sorted_imgs[i+1]}"] = {
            "pts1": pts1,
            "pts2": pts2
        }

    # Save
    with open(save_path, 'wb') as f:
        pickle.dump(all_matches, f)

    print(f"Saved SIFT matches for {canyon_name}_{folder_name} in {save_path}")
    return save_path


def get_canyon_sift_features_path(imgs_folder):
    '''
    returns path to the canyons sift features
    '''
    canyon_name = os.path.basename(os.path.dirname(os.path.normpath(imgs_folder)))
    folder_name = os.path.basename(os.path.normpath(imgs_folder))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "canyons_sift_features")
    os.makedirs(save_folder, exist_ok=True)

    canyon_sift_features_path = os.path.join(save_folder, f'sift_features_{canyon_name}_{folder_name}.pkl')

    return canyon_sift_features_path


def get_saved_sift_features_points(imgs_folder, idx):
    """
    Loads the SIFT feature matches for an image pair (idx-1, idx) from the precomputed pickle file.

    Args:
        imgs_folder (str): Path to the folder containing the images
        idx (int): Index of the current frame
    
    Returns:
        pts1: np.ndarray (N, 2)
        pts2: np.ndarray (N, 2)
    """

    # get path to sift features matches
    save_path = get_canyon_sift_features_path(imgs_folder)

    with open(save_path, 'rb') as f:
        all_matches = pickle.load(f)

    sorted_imgs = sorted(os.listdir(imgs_folder))

    # check if idx is valid
    if idx >= len(sorted_imgs):
        raise IndexError(f"idx must be between 0 and {len(sorted_imgs)-1}, got {idx}.")

    key = f"{sorted_imgs[idx-1]}-{sorted_imgs[idx]}"

    if idx == 0:
        key = f"{sorted_imgs[0]}-{sorted_imgs[1]}"

    pts1 = all_matches[key]['pts1']
    pts2 = all_matches[key]['pts2']

    return pts1, pts2


# --------------- functions for Priors generation -----------------------

def get_nearest_neighbor_prior(imgs_path, depth_path, idx):
    """
    Memory-efficient nearest-neighbor depth prior using a KD-tree.
    """

    # Load depth map
    depth_map_path = os.path.join(depth_path, sorted(os.listdir(depth_path))[idx])
    depth_map = tifffile.imread(depth_map_path).astype(np.float32)
    H, W = depth_map.shape

    # Get SIFT keypoints
    _, pts2_imgs = get_saved_sift_features_points(imgs_path, idx)
    if pts2_imgs.shape[0] == 0:
        print(f"Warning: No SIFT features found for {imgs_path}, frame {idx}. Returning zero prior.")
        return np.zeros((H, W), dtype=np.float32)
    
    xs = np.clip(pts2_imgs[:, 0].astype(int), 0, W-1)
    ys = np.clip(pts2_imgs[:, 1].astype(int), 0, H-1)
    depths_at_pts_imgs = depth_map[ys, xs]

    # KD-tree for keypoints
    keypoints = np.stack([ys, xs], axis=1)
    tree = cKDTree(keypoints)

    # Downsample grid for KD-tree query
    scale = 0.2
    Hq, Wq = int(H*scale), int(W*scale)
    yy, xx = np.meshgrid(np.linspace(0, H-1, Hq), np.linspace(0, W-1, Wq), indexing='ij')
    all_coords = np.stack([yy.ravel(), xx.ravel()], axis=1)

    # Query nearest keypoint
    _, nearest_idx = tree.query(all_coords)
    nn_prior_small = depths_at_pts_imgs[nearest_idx].reshape(Hq, Wq)

    # Upsample back to full resolution
    nn_prior = cv2.resize(nn_prior_small, (W, H), interpolation=cv2.INTER_NEAREST)

    return nn_prior


def get_gaussian_prior(imgs_path, idx, sigma=25):
    """
    Computes a Gaussian probability map at SIFT keypoints using a local window for efficiency.
    """

    # Load image to get size
    img_path = os.path.join(imgs_path, sorted(os.listdir(imgs_path))[idx])
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    # Get SIFT keypoints
    _, pts2 = get_saved_sift_features_points(imgs_path, idx)
    xs = np.clip(pts2[:, 0].astype(int), 0, W-1)
    ys = np.clip(pts2[:, 1].astype(int), 0, H-1)

    prob_map = np.zeros((H, W), dtype=np.float32)
    radius = 3*sigma

    for x, y in zip(xs, ys):
        # Define local window around keypoint
        x_min = max(x - radius, 0)
        x_max = min(x + radius + 1, W)
        y_min = max(y - radius, 0)
        y_max = min(y + radius + 1, H)

        # Create local coordinates
        X, Y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))

        # Gaussian
        gauss = np.exp(-((X - x)**2 + (Y - y)**2) / (2*sigma**2))

        # Update probability map using local max
        prob_map[y_min:y_max, x_min:x_max] = np.maximum(prob_map[y_min:y_max, x_min:x_max], gauss)

    return prob_map


def save_canyons_priors(imgs_path, depth_path, sigma=25):
    """
    Precomputes and saves nearest-neighbor and Gaussian priors for canyon datasets
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    canyons_priors = os.path.join(script_dir, "canyons_priors")
    canyon = os.path.basename(os.path.dirname(os.path.normpath(imgs_path)))
    basename = os.path.basename(imgs_path)

    os.makedirs(canyons_priors, exist_ok=True)

    nn_priors_path = os.path.join(canyons_priors, f"{canyon}_{basename}_nn_priors")
    gauss_priors_path = os.path.join(canyons_priors, f"{canyon}_{basename}_gauss_priors")

    os.makedirs(nn_priors_path, exist_ok=True)
    os.makedirs(gauss_priors_path, exist_ok=True)

    sorted_imgs = sorted(os.listdir(imgs_path))

    for idx in tqdm(range(len(sorted_imgs)), desc=f"Processing {os.path.basename(imgs_path)}", ncols = 100, leave= True):
        nn_prior = get_nearest_neighbor_prior(imgs_path, depth_path, idx)
        gauss_prior = get_gaussian_prior(imgs_path, idx, sigma=sigma)

        # Save priors
        nn_path = os.path.join(nn_priors_path, f"nn_{idx:05d}.npy")
        gauss_path = os.path.join(gauss_priors_path, f"gauss_{idx:05d}.npy") 
        np.save(nn_path, nn_prior)
        np.save(gauss_path, gauss_prior)


# --------------- functions for video generation -----------------------

def create_sift_video(imgs_folder, fps=10, codec = "XVID", ext=".avi"):
    """
    Create a video showing SIFT matches across a dataset.
    """

    # save path
    parent_folder_name = os.path.basename(os.path.dirname(os.path.normpath(imgs_folder)))
    folder_name = os.path.basename(os.path.normpath(imgs_folder))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, f"sift_videos_{ext.lstrip('.')}")
    os.makedirs(save_folder, exist_ok=True)

    name = f"sift_video_{parent_folder_name}_{folder_name}{ext}"
    save_path = os.path.join(save_folder, name)

    # check if video already exists
    if os.path.exists(save_path):
        print(f"-> Video for {name} already exists at {save_path} ->  Skipped")
        return save_path


    # load matches
    matches_file = save_all_matched_sift_features_of_dataset(imgs_folder)
    with open(matches_file, 'rb') as f:
        all_matches = pickle.load(f)

    # sort images
    sorted_imgs = sorted(os.listdir(imgs_folder))

    # get video size from first image
    H, W = cv2.imread(os.path.join(imgs_folder, sorted_imgs[0])).shape[:2]

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    # loop through consecutive images
    for idx  in tqdm(range(len(sorted_imgs)-1), desc="Progress", ncols = 100, leave= True):
        #img1_path = os.path.join(imgs_folder, sorted_imgs[idx])
        img2_path = os.path.join(imgs_folder, sorted_imgs[idx+1])

        # img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        key = f"{sorted_imgs[idx]}-{sorted_imgs[idx+1]}"
        if key not in all_matches:
            continue  # skip if no matches

        #pts1 = all_matches[key]['pts1']
        pts2 = all_matches[key]['pts2']

        # copy image for visualization
        vis = img2.copy()

        # draw points on second image
        for (x, y) in pts2:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)

        out.write(vis)

    out.release()
    print(f"Sift Video saved at: {save_path}")

    return save_path


def combine_sift_videos_side_by_side(canyon_path, fps=10, codec = "XVID", ext=".avi"):
    """
    combines the imgs seaErra sift videos horizontaly
    """

    # save path
    canyon_name = os.path.basename(os.path.normpath(canyon_path))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, f"sift_videos_{ext.lstrip('.')}")
    os.makedirs(save_dir, exist_ok=True)

    name = f"sift_video_{canyon_name}_imgs_seaErra{ext}"
    save_path = os.path.join(save_dir, name)

    # check if video already exists
    if os.path.exists(save_path):
        print(f"-> {name} already exists at {save_path} ->  Skipped")
        return save_path

    imgs_video = os.path.join(save_dir, f"sift_video_{canyon_name}_imgs{ext}")
    seaErra_video = os.path.join(save_dir, f"sift_video_{canyon_name}_seaErra{ext}")

    caps = [cv2.VideoCapture(v) for v in [imgs_video, seaErra_video]]

    # Get the height and width from the first video
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, (W*2, H))

    frame_count = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Combining videos for {canyon_name}:")
    for _ in tqdm(range(frame_count), desc="Progress", ncols = 100, leave= True):
        frames = []

        for cap in caps:
            _, frame = cap.read()
            frames.append(frame)
        
        combined_frame = np.hstack(frames)
        out.write(combined_frame)

    for cap in caps:
        cap.release()
    out.release()

    print(f"Saved combined video: {save_path}")

    return save_path



# ----------------------------------------------------------------------------------------
#                               Main (sift.py)
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":


    # ----- get sift features of Canyon Datasets ------------------------------------------

    print_title("sift.py")

    print("\nGetting sift features for the canyons imgs and seaErra datasets:\n")

    for path in tqdm(canyon_imgs_and_seaErra_paths, desc="Progress", ncols = 100, leave= True):
        sift_matches_tiny_canyon_imgs_path = save_all_matched_sift_features_of_dataset(path)


    # ----- Save Priors for all canyons ----------------------------------------------------

    print("\nSaving Priors for all canyon datasets:\n")

    for imgs_path, seaErra_path, depth_path in tqdm(canyon_imgs_seaErra_depth_paths, desc="Progress", ncols = 100, leave= True):
        save_canyons_priors(imgs_path, depth_path, sigma=25)
        save_canyons_priors(seaErra_path, depth_path, sigma=25)

    # ----- create videos  ------------------------------------------------------------------

    print("\nCreating videos of canyons imgs and seaErra with the sift features:\n")

    for path in tqdm(canyon_imgs_and_seaErra_paths, desc="Progress", ncols = 100, leave= True):
        create_sift_video(path, fps=10, codec = "mp4v", ext=".mp4")

    print("\nCreating combinded videos of canyons imgs and seaErra with the sift features:\n")

    for path in tqdm(canyons_paths, desc="Progress", ncols = 100, leave= True):
        combine_sift_videos_side_by_side(path, fps=10, codec = "mp4v", ext=".mp4")

