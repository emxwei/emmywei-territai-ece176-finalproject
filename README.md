# ECE 176 Final Project - Brain Tumor Segmentation
**Team Members:** Emmy Wei & Terri Tai  
**Course:** ECE 176 - Deep Learning  

## Project Overview
Accurate localization of brain tumors in medical images is critical for diagnosis and surgical guidance. In this project, we implemented a **2D U-Net Convolutional Neural Network** from scratch to perform automated brain tumor segmentation on multimodal MRI scans (T1, T1CE, T2, and FLAIR). 

To handle the severe class imbalance between healthy tissue and diffuse tumor boundaries, the network is optimized using a custom **Dice Loss** function.

## Repository Structure
To enable concurrent development and clean version control, the project codebase is modularized into the following files:

* `dataset.py`: Handles loading the BraTS multimodal MRI slices (HDF5/NIfTI), applying augmentations, and formatting the 4-channel inputs and multi-label ground truth masks.
* `model.py`: Contains the PyTorch implementation of the 4-level 2D U-Net architecture, including the encoder, bottleneck, decoder, and skip connections.
* `loss.py`: Contains our custom implementation of the Dice Loss function for multi-label segmentation optimization.
* `train.py`: Contains the standard PyTorch training and validation loops, along with metric tracking (Loss and Dice Score).
* `main_notebook.ipynb`: The main control center. Imports all the modules above, initializes the dataloaders, runs the training loop, and visualizes the results.

## Environment & Requirements
Ensure you have the following libraries installed in your environment (Datahub/Local) before running:
```sh
pip install torch torchvision numpy matplotlib h5py tqdm nibabel
```

## Prepare Datasets
This project uses the publicly available **BraTS (Brain Tumor Segmentation) Dataset**. 

Before running the code, you must download the dataset and place it in the correct directory. 
1. Download the BraTS 2020/2021 dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) or the Cancer Imaging Archive.
2. Extract the dataset into a folder named `data/` in the root directory of this project.

Your directory should look like this:
```text
emmywei-territai-ece176-finalproject/
├── data/                  # <-- Place extracted BraTS data here
├── dataset.py
├── model.py
├── loss.py
├── train.py
└── main_notebook.ipynb
```

## Implementation & How to Run

You should run all code blocks inside the Jupyter Notebook. It is highly recommended to run this on a machine with a dedicated GPU (e.g., UCSD Datahub).

1. Open `main_notebook.ipynb`.
2. Ensure the `DEVICE` in Cell 2 successfully detects your `cuda` GPU.
3. Update the dataset file paths in Cell 3 to point to your local `data/` folder.
4. **Run All Cells** to initialize the U-Net, train the model over the specified epochs, and generate the side-by-side MRI and prediction visualizations.
```

***