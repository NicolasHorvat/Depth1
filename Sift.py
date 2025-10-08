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

    if des1 is None or des2 is None:
        return np.array([]), np.array([])
    
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
    print(f"Computing Sift matches for {parent_folder_name}_{folder_name}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "datasets_sift_matches")
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, f'sift_matches_{parent_folder_name}_{folder_name}.pkl')

    # check if file already exists
    if os.path.exists(save_path):
        print(f"-> Matches for {parent_folder_name}_{folder_name} already exist at {save_path} ->  Skipping computation")
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
    matches_file = save_all_sift_features_of_dataset(imgs_folder)
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


# ----------------------------Main------------------------------------------------------------

# ----------------- get sift features of Canyon Datasets -------------------------------------

for path in tqdm(canyon_imgs_and_seaErra_paths, desc="Progress", ncols = 100, leave= True):
    sift_matches_tiny_canyon_imgs_path = save_all_sift_features_of_dataset(path)

# ------------------- create a video of it ---------------------------------------------------

for path in tqdm(canyon_imgs_and_seaErra_paths, desc="Progress", ncols = 100, leave= True):
    create_sift_video(path, fps=10, codec = "mp4v", ext=".mp4")


for path in tqdm(canyons_paths, desc="Progress", ncols = 100, leave= True):
    combine_sift_videos_side_by_side(path, fps=10, codec = "mp4v", ext=".mp4")


# ---------------- image visualization -------------------------------------------------------
sorted_tiny_canyon_imgs = sorted(os.listdir(tiny_canyon_imgs_path))
img1_path = os.path.join(tiny_canyon_imgs_path, sorted_tiny_canyon_imgs[500])
img2_path = os.path.join(tiny_canyon_imgs_path, sorted_tiny_canyon_imgs[501])

sorted_tiny_canyon_seaErra = sorted(os.listdir(tiny_canyon_seaErra_path))
img3_path = os.path.join(tiny_canyon_seaErra_path, sorted_tiny_canyon_seaErra[500])
img4_path = os.path.join(tiny_canyon_seaErra_path, sorted_tiny_canyon_seaErra[501])

visualize_sift_matches(img1_path, img2_path)
visualize_sift_matches(img3_path, img4_path)

# way more sift features detected in seaErra set than in imgs set

