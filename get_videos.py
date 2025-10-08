import os
import cv2
import numpy as np
import tifffile
from tqdm import tqdm

from paths import *

'''
script used to create videos of the FLSea canyons Datasets
'''


# ---------- Make video ----------    make_video(path, codec = default_codec, ext = default_ext)
def make_video(imgs_path, fps=10, codec = "XVID", ext=".avi"):
    '''
    Creates a Video of the sorted images in the specified folder.
    Skipps creation if it already exists
    Returns the Path.
    '''

    # save path
    parent_dir_name = os.path.basename(os.path.dirname(os.path.normpath(imgs_path)))
    dir_name = os.path.basename(os.path.normpath(imgs_path))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, f"videos_{ext.lstrip('.')}")
    os.makedirs(save_dir, exist_ok=True)

    name = f"video_{parent_dir_name}_{dir_name}{ext}"
    save_path = os.path.join(save_dir, name)

    # check if video already exists
    if os.path.exists(save_path):
        print(f"-> {name} already exists at {save_path} ->  Skipped")
        return save_path
    
    
    sorted_imgs = sorted(os.listdir(imgs_path))
    
    # cv.imread only works for tiff but not tif so this is needed
    img = tifffile.imread(os.path.join(imgs_path, sorted_imgs[0]))
    if len(img.shape) == 2:  # grayscale
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    
    print(f"Making: {name}")
    for idx in tqdm(range(len(sorted_imgs)), desc="Progress", ncols = 100, leave= True):
        img_path = os.path.join(imgs_path, sorted_imgs[idx])
        img = tifffile.imread(img_path)
        if len(img.shape) == 2:  # grayscale
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out.write(img)

    out.release()

    print(f"Saved video at: {save_path}")

    return save_path


def combine_videos_side_by_side(canyon_path, fps=10, codec = "XVID", ext=".avi"):
    """
    combines the imgs seaErra and depth videos horizontaly
    """

    # save path
    canyon_name = os.path.basename(os.path.normpath(canyon_path))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, f"videos_{ext.lstrip('.')}")
    os.makedirs(save_dir, exist_ok=True)

    name = f"video_{canyon_name}_imgs_seaErra_depth{ext}"
    save_path = os.path.join(save_dir, name)

    # check if video already exists
    if os.path.exists(save_path):
        print(f"-> {name} already exists at {save_path} ->  Skipped")
        return save_path

    imgs_video = os.path.join(save_dir, f"video_{canyon_name}_imgs{ext}")
    seaErra_video = os.path.join(save_dir, f"video_{canyon_name}_seaErra{ext}")
    depth_video = os.path.join(save_dir, f"video_{canyon_name}_depth{ext}")

    caps = [cv2.VideoCapture(v) for v in [imgs_video, seaErra_video, depth_video]]

    # Get the height and width from the first video
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, (W*3, H))

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


def stack_all_combined_videos_vertically(fps=10, codec="XVID", ext=".avi"):
    """
    Combine all canyon videos vertically, handling different lengths.
    """
    import os, cv2, numpy as np
    from tqdm import tqdm

    # Save path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, f"videos_{ext.lstrip('.')}")
    os.makedirs(save_dir, exist_ok=True)

    name = f"video_all_canyons_imgs_seaErra_depth{ext}"
    save_path = os.path.join(save_dir, name)

    if os.path.exists(save_path):
        print(f"-> {name} already exists -> Skipped")
        return save_path

    video_files = sorted([os.path.join(save_dir, f)
                          for f in os.listdir(save_dir)
                          if f.endswith(ext) and "imgs_seaErra_depth" in f])

    caps = [cv2.VideoCapture(v) for v in video_files]

    # Get width and height from first video
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the max frame count across all videos
    frame_count = max([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H * len(caps)))

    print(f"Stacking the combined videos:")
    for _ in tqdm(range(frame_count), desc="Progress", ncols=100, leave=True):
        frames = []

        for cap in caps:
            ret, frame = cap.read()

            if not ret:
                frame = np.zeros((H, W, 3), dtype=np.uint8)

            frames.append(frame)

        combined_frame = np.vstack(frames)
        out.write(combined_frame)

    for cap in caps:
        cap.release()
    out.release()

    print(f"Saved stacked video: {save_path}")
    return save_path


# ---------------------------------------------------------------------------------
#                  Main - Calling Functions
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Set default codec and extension
    if False:
        default_codec = "XVID"  # for AVI
        default_ext   = ".avi"
    if True:
        default_codec = "mp4v"   # for mp4
        default_ext = ".mp4"

    for path in canyon_imgs_seaErra_and_depth_paths:

        make_video(path, fps=10, codec = default_codec, ext = default_ext)

    for path in canyons_paths:
        combine_videos_side_by_side(path, fps = 10, codec = default_codec, ext = default_ext)

    stack_all_combined_videos_vertically(fps = 10, codec = default_codec, ext = default_ext)