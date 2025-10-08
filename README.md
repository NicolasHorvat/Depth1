# Underwater Depth Prediction – FLSea Canyons

This project tackles **underwater depth prediction** using the **FLSea Canyons Dataset** ([Kaggle link](https://www.kaggle.com/datasets/viseaonlab/flsea-vi)).  
It is inspired by [this research](https://ieeexplore.ieee.org/abstract/document/10611007).

-> The Canyons Dataset is **not included** in this repository. Download it from Kaggle and specify the path in the scripts.

---

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Main script to select and run different tasks. |
| `paths.py` | Paths to Canyon datasets and path lists |
| `dataset.py` | Dataset classes (with or without prior information like SIFT features). |
| `model.py` | Model definitions (currently includes U-Nets). |
| `train.py` | Training function for models. |
| `test.py` | Testing and evaluation scripts. |
| `utils.py` | Utility functions used throughout the project. |
| `log_plot.py` | Plots training logs and compares metrics like time and loss across runs. |
| `sift_test.py` | Experimental SIFT feature detection tests. |
| `Sift.py` | SIFT feature matches across two framesgeneration. And Video |
| `get_videos.py` | Generates videos of the dataset (RGB, SeaError, Depth). |

---

## Getting Started

1. **Download the dataset** from Kaggle.  
2. Update the **dataset path** in paths.py:  
3. pip install -r requirements.txt (into your Venv)
4. experiment in main.py, Sift.py ect.

