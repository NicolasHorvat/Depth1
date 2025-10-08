import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
 

# --------------------------------------------------------------------------------- #
#                                    Paths
# --------------------------------------------------------------------------------- #

# reminder to change the canyons folder path
print("\n(Reminder to change the canyons folder path)")

# canyons_path = r"C:\Users\nicol\ETH\Master_Thesis\canyons" # path to canyons data from FLSea VI
canyons_path = r"H:\canyons" # path to canyons data from FLSea VI on my SSD                                    <-------  Dataset Path !!!

tiny_canyon_path = os.path.join(canyons_path, "tiny_canyon")
tiny_canyon_depth_path = os.path.join(tiny_canyon_path, "depth")
tiny_canyon_imgs_path = os.path.join(tiny_canyon_path, "imgs")
tiny_canyon_seaErra_path = os.path.join(tiny_canyon_path, "seaErra")




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
    Returns the path if it already exists
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
        print(f"Matches for {folder_name} already exist at {save_path}. Skipping computation.")
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



def create_sift_video(matches_file, imgs_folder, fps=10):
    """
    Create a video showing SIFT matches across a dataset.
    """
    # load matches
    with open(matches_file, 'rb') as f:
        all_matches = pickle.load(f)

    # sort images
    sorted_imgs = sorted(os.listdir(imgs_folder))

    # determine save path
    dataset_name = os.path.basename(os.path.normpath(imgs_folder))
    save_name = f"sift_video_{dataset_name}.mp4"
    save_path = os.path.join(os.path.dirname(matches_file), save_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "sift_videos")
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, save_name)

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




# ----------------------------Main--------------------------------

# ----------------- get sift features of dataset -------------
sift_matches_tiny_canyon_imgs_path = save_all_sift_features_of_dataset(tiny_canyon_imgs_path)

# ------------------- create a video of it -------------------
create_sift_video(sift_matches_tiny_canyon_imgs_path, tiny_canyon_imgs_path, fps=10)


# ---------------- image visualization --------------
sorted_tiny_canyon_imgs = sorted(os.listdir(tiny_canyon_imgs_path))
img1_path = os.path.join(tiny_canyon_imgs_path, sorted_tiny_canyon_imgs[500])
img2_path = os.path.join(tiny_canyon_imgs_path, sorted_tiny_canyon_imgs[501])

visualize_sift_matches(img1_path, img2_path, max_matches=100)

