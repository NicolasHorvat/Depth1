import os

# ------------------------------------------------------------------------
#                      Set the Canyons dataset path
# ------------------------------------------------------------------------

canyons_path = r"H:\canyons"  # <---- change to your dataset location!!!


# ------------------------ Flatiron Canyon -------------------------------

flatiron_path = os.path.join(canyons_path, "flatiron")
flatiron_depth_path = os.path.join(flatiron_path, "depth")
flatiron_imgs_path = os.path.join(flatiron_path, "imgs")
flatiron_seaErra_path = os.path.join(flatiron_path, "seaErra")


# -------------------------- Horse Canyon --------------------------------

horse_canyon_path = os.path.join(canyons_path, "horse_canyon")
horse_canyon_depth_path = os.path.join(horse_canyon_path, "depth")
horse_canyon_imgs_path = os.path.join(horse_canyon_path, "imgs")
horse_canyon_seaErra_path = os.path.join(horse_canyon_path, "seaErra")


# -------------------------- Tiny Canyon ---------------------------------

tiny_canyon_path = os.path.join(canyons_path, "tiny_canyon")
tiny_canyon_depth_path = os.path.join(tiny_canyon_path, "depth")
tiny_canyon_imgs_path = os.path.join(tiny_canyon_path, "imgs")
tiny_canyon_seaErra_path = os.path.join(tiny_canyon_path, "seaErra")


# ---------------------------- U-Canyon ----------------------------------

u_canyon_path = os.path.join(canyons_path, "u_canyon")
u_canyon_depth_path = os.path.join(u_canyon_path, "depth")
u_canyon_imgs_path = os.path.join(u_canyon_path, "imgs")
u_canyon_seaErra_path = os.path.join(u_canyon_path, "seaErra")

# ------------------------------------------------------------------------
#                                   Lists
# ------------------------------------------------------------------------

# Canyons_path
canyons_paths = [
    flatiron_path,
    horse_canyon_path,
    tiny_canyon_path,
    u_canyon_path
]

# RGB images
canyon_imgs_paths = [
    flatiron_imgs_path,
    horse_canyon_imgs_path,
    tiny_canyon_imgs_path,
    u_canyon_imgs_path
]

# SeaErra images
canyon_seaErra_paths = [
    flatiron_seaErra_path,
    horse_canyon_seaErra_path,
    tiny_canyon_seaErra_path,
    u_canyon_seaErra_path
]

# rgb and seaErra paths
canyon_imgs_and_seaErra_paths = [
    flatiron_imgs_path,
    horse_canyon_imgs_path,
    tiny_canyon_imgs_path,
    u_canyon_imgs_path,
    flatiron_seaErra_path,
    horse_canyon_seaErra_path,
    tiny_canyon_seaErra_path,
    u_canyon_seaErra_path
]

# rgb, seaErra and depth paths
canyon_imgs_seaErra_and_depth_paths = [
    flatiron_imgs_path,
    horse_canyon_imgs_path,
    tiny_canyon_imgs_path,
    u_canyon_imgs_path,
    flatiron_seaErra_path,
    horse_canyon_seaErra_path,
    tiny_canyon_seaErra_path,
    u_canyon_seaErra_path,
    flatiron_depth_path,
    horse_canyon_depth_path,
    tiny_canyon_depth_path,
    u_canyon_depth_path
]

# ------------------------------------------------------------------------
#               Lists of image/depth or seaErra/depth pairs
# ------------------------------------------------------------------------

canyon_imgs_seaErra_depth_paths = [
    (flatiron_imgs_path, flatiron_seaErra_path, flatiron_depth_path),
    (horse_canyon_imgs_path, horse_canyon_seaErra_path, horse_canyon_depth_path),
    (tiny_canyon_imgs_path, tiny_canyon_seaErra_path, tiny_canyon_depth_path),
    (u_canyon_imgs_path, u_canyon_seaErra_path, u_canyon_depth_path)
]

canyon_imgs_depth_paths = [
    (flatiron_imgs_path, flatiron_depth_path),
    (horse_canyon_imgs_path, horse_canyon_depth_path),
    (tiny_canyon_imgs_path, tiny_canyon_depth_path),
    (u_canyon_imgs_path, u_canyon_depth_path)
]

canyon_seaErra_depth_paths = [
    (flatiron_seaErra_path, flatiron_depth_path),
    (horse_canyon_seaErra_path, horse_canyon_depth_path),
    (tiny_canyon_seaErra_path, tiny_canyon_depth_path),
    (u_canyon_seaErra_path, u_canyon_depth_path)
]
