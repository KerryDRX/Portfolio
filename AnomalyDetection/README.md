<div align="center">

  # Medical Image Anomaly Detection

</div>

## Overview
- Unsupervised anomaly detection of computed tomography (CT) and ultra-widefield (UWF) images, with anomaly defined by
    - head CT images: if hemorrhage exists
    - fundus UFW images: if image artifacts hinder detection of sea-fan neovascularization
- Code of two tasks is in `CT` and `UWF` folders, respectively.

## CT Anomaly Detection

### Step 1: Preparation

Store images in: `<data_dir>/train/<good_label>/*`, `<data_dir>/test/<good_label>/*`, and `<data_dir>/test/<bad_label>/*`. Set parameters in `config.py`.

### Step 2: Model Training

Run notebook to train the model, e.g., `HeadCT_fAnoGAN.ipynb`.

### Step 3: Result Visualization

Run notebook to check results, e.g., `visualization_fAnoGAN.ipynb`.

## UWF Anomaly Detection

### Step 1: Preparation

In `config.py`, set `PATHS.DATA_DIR` to the data directory (containing two subfolders of good and poor images) and set `PATHS.OUTPUT_DIR` to the output directory. UWF fundus images of two classes are respectively stored in `PATHS.DATA_DIR/Good` and `PATHS.DATA_DIR/Poor`.

### Step 2: Model Training
```
python train.py
```
All the results will be saved to `PATHS.DATA_DIR`.

### Step 3: Model Testing

Run the notebook `test.ipynb` to test model performance.
