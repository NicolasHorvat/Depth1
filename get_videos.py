import os
import cv2
import numpy as np
import tifffile
from tqdm import tqdm

'''
script used to create videos of the FLSea canyons Datasets
'''

# ---------- Dataset Paths ----------
canyons_path = r"H:\canyons"  # Path to FLSea VI dataset  <-- change this to where you saved the canyons dataset

canyon_folders = {
    "flatiron": os.path.join(canyons_path, "flatiron"),
    "horse_canyon": os.path.join(canyons_path, "horse_canyon"),
    "tiny_canyon": os.path.join(canyons_path, "tiny_canyon"),
    "u_canyon": os.path.join(canyons_path, "u_canyon")
}

# Set default codec and extension
if False:
    default_codec = "XVID"  # for AVI
    default_ext   = ".avi"
if True:
    default_codec = "mp4v"   # for mp4
    default_ext = ".mp4"

# ---------- Videos folder ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
videos_dir = os.path.join(script_dir, f"videos_{default_ext.lstrip('.')}")
os.makedirs(videos_dir, exist_ok=True)


# ---------- Read TIFF helper ----------
def read_frame_tiff(frame_path):
    try:
        frame = tifffile.imread(frame_path)
        if frame is None:
            raise ValueError
        
        # Convert single-channel to 3-channel
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Normalize if not uint8
        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)
            
        return frame
    except Exception as e:
        print(f"[WARN] Failed to read {frame_path}, skipping.")
        return None


# ---------- Make video ----------
def make_video(folder_path, output_name, fps=10, codec = "XVID", ext=".avi"):
    
    img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))])
    if len(img_files) == 0:
        print(f"No images found in {folder_path}, skipping.")
        return
    
    # Read first valid frame to get size
    for f in img_files:
        first_frame = read_frame_tiff(os.path.join(folder_path, f))
        if first_frame is not None:
            break
    else:
        print(f"No readable frames in {folder_path}, skipping.")
        return
    
    H, W, _ = first_frame.shape
    output_name += ext
    video_path = os.path.join(videos_dir, output_name)
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    
    print(f"Writing {output_name}: ", end="")
    for img_file in tqdm(img_files):
        frame = read_frame_tiff(os.path.join(folder_path, img_file))
        if frame is not None:
            out.write(frame)
    out.release()
    print(f"Saved video: {video_path}")


def combine_videos_side_by_side(canyon_name, fps=10, codec = "XVID", ext=".avi"):
    rgb_vid = os.path.join(videos_dir, f"{canyon_name}_rgb{ext}")
    sea_vid = os.path.join(videos_dir, f"{canyon_name}_seaErra{ext}")
    depth_vid = os.path.join(videos_dir, f"{canyon_name}_depth{ext}")

    caps = [cv2.VideoCapture(v) for v in [rgb_vid, sea_vid, depth_vid]]

    # Check all opened correctly
    if not all([cap.isOpened() for cap in caps]):
        print(f"Error opening videos for {canyon_name}")
        return

    # Get the height and width from the first video
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    combined_path = os.path.join(videos_dir, f"{canyon_name}_combined{ext}")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(combined_path, fourcc, fps, (W*3, H))  # width is 3x for side-by-side

    frame_count = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Combining videos for {canyon_name}:")
    
    for _ in tqdm(range(frame_count)):
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((H, W, 3), dtype=np.uint8)  # blank frame if video ended early
            frames.append(frame)
        
        combined_frame = np.hstack(frames)  # horizontal stack
        out.write(combined_frame)

    for cap in caps:
        cap.release()
    out.release()
    print(f"Saved combined video: {combined_path}")


def stack_all_combined_videos_vertically(output_name="all_canyons_stacked.avi", codec = "XVID", ext = ".avi"):
    # Find all combined canyon videos
    combined_videos = sorted([
        os.path.join(videos_dir, f)
        for f in os.listdir(videos_dir)
        if f.endswith(f"_combined{ext}")
    ])
    
    if not combined_videos:
        print("⚠️ No combined videos found!")
        return

    print(f"Found {len(combined_videos)} combined videos:")
    for v in combined_videos:
        print("  -", os.path.basename(v))

    # Open all videos
    caps = [cv2.VideoCapture(v) for v in combined_videos]

    # Check successful openings
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error opening video {combined_videos[i]}")
            return

    # Use first video as reference
    width  = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = caps[0].get(cv2.CAP_PROP_FPS)

    # Determine longest video
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    max_frames = max(frame_counts)

    stacked_height = height * len(caps)
    output_path = os.path.join(videos_dir, output_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, (width, stacked_height))

    print(f"\nCreating {output_name} ({len(caps)} videos stacked vertically)...")

    last_frames = [None] * len(caps)

    for _ in tqdm(range(max_frames)):
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                # Repeat last available frame if video ended
                if last_frames[i] is not None:
                    frame = last_frames[i]
                else:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
            last_frames[i] = frame
            frames.append(frame)

        stacked_frame = np.vstack(frames)
        out.write(stacked_frame)

    # Cleanup
    for cap in caps:
        cap.release()
    out.release()

    print(f"Saved stacked video: {output_path}")


# ---------- Main ----------
for canyon_name, canyon_path in canyon_folders.items():
    print(f"\nProcessing canyon: {canyon_name}")
    
    rgb_path = os.path.join(canyon_path, "imgs")
    seaErra_path = os.path.join(canyon_path, "seaErra")
    depth_path = os.path.join(canyon_path, "depth")
    
    videos_to_make = [
        (rgb_path, f"{canyon_name}_rgb"),
        (seaErra_path, f"{canyon_name}_seaErra"),
        (depth_path, f"{canyon_name}_depth")
    ]
    
    for folder_path, output_name in videos_to_make:
        video_path = os.path.join(videos_dir, output_name + default_ext)
        if os.path.exists(video_path):
            print(f"Video {output_name} already exists, skipping...")
        else:
            make_video(folder_path, output_name, codec = default_codec, ext = default_ext)

    # ---------- Combine imgs, SeaError, and Depth ----------
    combined_video_path = os.path.join(videos_dir, f"{canyon_name}_combined{default_ext}")
    if os.path.exists(combined_video_path):
        print(f"Combined video {canyon_name}_combined{default_ext} already exists, skipping...")
    else:
        combine_videos_side_by_side(canyon_name, codec = default_codec, ext = default_ext)

# ---------- Stack all combined videos vertically ----------
stacked_video_path = os.path.join(videos_dir, f"all_canyons_stacked{default_ext}")
if os.path.exists(stacked_video_path):
    print(f"Stacked video already exists at {stacked_video_path}, skipping...")
else:
    stack_all_combined_videos_vertically(output_name = f"all_canyons_stacked{default_ext}", codec = default_codec, ext = default_ext)