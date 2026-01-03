# Data Folder

This folder should contain preprocessed PTB-XL data:

## Required Files (not in Git - too large)
- `X_train.npy` (2.0 GB)
- `X_val.npy` (0.4 GB)
- `X_test.npy` (0.4 GB)
- `y_train.npy`
- `y_val.npy`
- `y_test.npy`
- `scaler.pkl`

## How to Generate
Run the preprocessing notebooks in order:
1. `notebooks/1_data_loading_exploration.ipynb`
2. `notebooks/2_data_preprocessing.ipynb`

This will create all necessary `.npy` files in this folder.
