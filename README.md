# Underwater Depth Prediction – FLSea Canyons

This project tackles **underwater depth prediction** using the **FLSea Canyons Dataset** ([Kaggle link](https://www.kaggle.com/datasets/viseaonlab/flsea-vi)).  
It is inspired by [this research](https://ieeexplore.ieee.org/abstract/document/10611007).

-> The Canyons Dataset is **not included** in this repository. Download it from Kaggle and specify the path in the scripts.



## Project Structure ##

main.py       - main file (select what to run)
dataset.py    - Dataset classes (with or without prior information (sift), etc.)
model.py      - Different models for training (currently U-Nets)
train.py      - Training function
test.py       - Testing / evaluation function
utils.py      - Utility functions
get_videos.py - Generates videos of the dataset (RGB, SeaError, Depth)
log_plot.py   - Plots simple comparison of time/loss for different runs
sift_test.py  - Experimental SIFT detection tests
