import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
 
from paths import *


def match_sift_features(img1, img2):
    """
    Detects and matches SIFT features between two frames.
    """
    # Detect Keypoints and Compute Descriptors
    sift = cv2.SIFT_create()
    kps1, des1 = sift.detectAndCompute(img1, None)
    kps2, des2 = sift.detectAndCompute(img2, None)

    # Match Descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Outlier Rejection (Lowe)
    good = []
    for mn in matches:
        if len(mn) != 2:
            continue  # skip if less than 2 matches
        m, n = mn
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return np.array([]), np.array([])

    # get coordinates
    pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good])

    # RANSAC fundamental matrix filtering
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None:
        return np.array([]), np.array([])

    inliers = mask.ravel() == 1
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    return pts1_in, pts2_in


def visualize_sift_matches(img1_path, img2_path, max_matches=1000):
    """
    Visualizes SIFT matches between two frames.
    """

    # read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # get matching sift features
    pts1, pts2 = match_sift_features(img1, img2)

    # Combine
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2


    for i in range(min(len(pts1), max_matches)):
        # get points
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        p1 = (int(x1), int(y1))
        p2 = (int(x2) + w1, int(y2))

        # random color
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # draw circles and lines
        cv2.circle(vis, p1, 4, color, -1)
        cv2.circle(vis, p2, 4, color, -1)
        cv2.line(vis, p1, p2, color, 1)

    # plot
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)) # bgr -> rgb
    plt.title(f"SIFT matches: {len(pts1)} inliers shown")
    plt.axis('off')
    plt.show()


def save_all_sift_features_of_dataset(imgs_folder):
    '''
    creates and saves sift features for a given dataset
    saves them in the datasets_sift_matches folder
    Returns the path
    '''

    # save path
    parent_folder_name = os.path.basename(os.path.dirname(os.path.normpath(imgs_folder)))
    folder_name = os.path.basename(os.path.normpath(imgs_folder))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "datasets_sift_matches")
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, f'sift_matches_{parent_folder_name}_{folder_name}.pkl')

    # check if file already exists
    if os.path.exists(save_path):
        print(f"-> Matches for {folder_name} already exist at {save_path} ->  Skipping computation")
        return save_path 

    sorted_imgs = sorted(os.listdir(imgs_folder))

    all_matches = {}

    for i in tqdm(range(len(sorted_imgs)-1), desc="Progress", ncols = 100, leave= True):
        img1_path = os.path.join(imgs_folder, sorted_imgs[i])
        img2_path = os.path.join(imgs_folder, sorted_imgs[i+1])

        # read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # compute matches
        pts1, pts2 = match_sift_features(img1, img2)

        # store in dictionary
        all_matches[f"{sorted_imgs[i]}-{sorted_imgs[i+1]}"] = {
            "pts1": pts1,
            "pts2": pts2
        }
    

    with open(save_path, 'wb') as f:
        pickle.dump(all_matches, f)

    print(f"Saved Sift matches for {parent_folder_name}_{folder_name} in {save_path}")

    return save_path


def create_sift_video(imgs_folder, fps=10):
    """
    Create a video showing SIFT matches across a dataset.
    """

    # save path
    parent_folder_name = os.path.basename(os.path.dirname(os.path.normpath(imgs_folder)))
    folder_name = os.path.basename(os.path.normpath(imgs_folder))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "sift_videos")
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, f"sift_video_{parent_folder_name}_{folder_name}.mp4")

    # check if video already exists
    if os.path.exists(save_path):
        print(f"-> Video for {parent_folder_name}_{folder_name} already exists at {save_path} ->  Skipping creation")
        return save_path


    # load matches
    matches_file = save_all_sift_features_of_dataset(imgs_folder)
    with open(matches_file, 'rb') as f:
        all_matches = pickle.load(f)

    # sort images
    sorted_imgs = sorted(os.listdir(imgs_folder))

    # get video size from first image
    frame_height, frame_width = cv2.imread(os.path.join(imgs_folder, sorted_imgs[0])).shape[:2]

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

    # loop through consecutive images
    for i in tqdm(range(len(sorted_imgs)-1), desc="Progress", ncols = 100, leave= True):
        img1_path = os.path.join(imgs_folder, sorted_imgs[i])
        img2_path = os.path.join(imgs_folder, sorted_imgs[i+1])

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        key = f"{sorted_imgs[i]}-{sorted_imgs[i+1]}"
        if key not in all_matches:
            continue  # skip if no matches

        pts1 = all_matches[key]['pts1']
        pts2 = all_matches[key]['pts2']

        # copy image for visualization
        vis = img2.copy()

        # draw points on second image
        for (x, y) in pts2:
            cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 255), -1)

        # draw lines to first image
        #for (x1, y1), (x2, y2) in zip(pts1, pts2):
        #    cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)

        out.write(vis)

    out.release()
    print(f"Sift Video saved to {save_path}")

    return save_path



# ----------------------------Main-----------------------------------------------------------------

# ----------------- get sift features of Canyon Datasets ------------------------------------------

for path in tqdm(canyon_imgs_and_seaErra_paths, desc="Progress", ncols = 100, leave= True):

    sift_matches_tiny_canyon_imgs_path = save_all_sift_features_of_dataset(path)

# ------------------- create a video of it ---------------------------------------------------------
for path in tqdm(canyon_imgs_and_seaErra_paths, desc="Progress", ncols = 100, leave= True):
    
    create_sift_video(path, fps=10)


# ---------------- image visualization -------------------------------------------------------------
sorted_tiny_canyon_imgs = sorted(os.listdir(tiny_canyon_imgs_path))
img1_path = os.path.join(tiny_canyon_imgs_path, sorted_tiny_canyon_imgs[500])
img2_path = os.path.join(tiny_canyon_imgs_path, sorted_tiny_canyon_imgs[501])

visualize_sift_matches(img1_path, img2_path)

