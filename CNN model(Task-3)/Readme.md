# Land Use Classification with Custom CNN (EuroSAT RGB)

## 1. Project Overview

This project trains and compares two convolutional neural networks (a baseline CNN and a deeper custom CNN) for **land use / land cover classification** on the EuroSAT RGB satellite image dataset.  
The notebook covers dataset preparation, manual train/validation/test splitting, model training with regularization and early stopping, and evaluation with accuracy, a classification report, and a confusion matrix.

---

## 2. Dataset (EuroSAT from Zenodo)

- Dataset: **EuroSAT – Land Use and Land Cover Classification with Sentinel-2**  
- Download (Zenodo): https://zenodo.org/records/7711810  
- Variant used: `EuroSAT_RGB.zip` – RGB version with R, G, B bands as JPEG images.

Key characteristics:

- Around 27,000 geo‑referenced image patches.  
- Image size: **64 × 64 pixels**.  
- 10 land use / land cover classes, for example: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, Sea/Lake.

The notebook assumes that `EuroSAT_RGB` has been downloaded and extracted, and points to this directory as the raw dataset root.

---

## 3. Dataset Preparation

### 3.1 Directory layout and splitting

The raw EuroSAT RGB directory contains one subfolder per class with JPEG patches.  
The notebook:

- Defines `original_root` as the path to this RGB directory.  
- Creates `EuroSAT_RGB_split` with three subfolders:
  - `train/`
  - `val/`
  - `test/`
- For each class:
  - Lists all image files and shuffles them with a fixed random seed.  
  - Splits into:
    - 70% training
    - 15% validation
    - 15% test  
  - Copies files into `train/class_name`, `val/class_name`, `test/class_name`.

This produces a clean, reproducible folder structure ready for `ImageDataGenerator.flow_from_directory`.

### 3.2 Global configuration

The notebook centralizes:

- `train_dir`, `val_dir`, `test_dir` paths.  
- Image size: **64 × 64** (EuroSAT patch size).  
- Color mode: RGB (3 channels).  
- Batch size (for example, 64).

---

## 4. Data Loading and Augmentation

Keras `ImageDataGenerator` is used to stream images from disk.

- **Training generator**:
  - Rescales pixel values (e.g., divide by 255).  
  - Applies data augmentation (random flips, slight rotations, etc.).

- **Validation and test generators**:
  - Rescale images only (no augmentation).

All generators:

- Read from the split directories.  
- Resize images to 64 × 64.  
- Use `class_mode="sparse"` and `color_mode="rgb"`.  
- Infer class indices from subfolder names and store the sorted class list.

---

## 5. Training Utilities

### 5.1 Class names

- Collects class folders under `train_dir`.  
- Filters out any non‑class folders.  
- Sorts the class names and uses this fixed order for label indices.

### 5.2 Early stopping

- `EarlyStopping` callback:
  - `monitor="val_loss"`  
  - `patience` set to a few epochs  
  - `restore_best_weights=True`

This stops training when validation loss no longer improves and restores the best model weights.

---

## 6. Baseline CNN

### 6.1 Architecture

Compact CNN used as a reference:

- Input: 64 × 64 × 3 RGB.  
- Two convolutional blocks, each:
  - Conv2D (moderate filters, small kernel)  
  - Batch Normalization  
  - LeakyReLU  
  - MaxPooling2D  
- Flatten (or equivalent).  
- Dense(128) with LeakyReLU and dropout.  
- Output Dense(10) with softmax.

L2 regularization is applied to convolutional and dense layers.

### 6.2 Compilation and training

- Loss: `sparse_categorical_crossentropy`.  
- Metric: accuracy.  
- Optimizer: Adam (standard learning rate).  
- Training:
  - Uses training generator, validates on validation generator.  
  - Maximum epochs (e.g., 20) with early stopping.  
  - Stores loss and accuracy history for plotting.

---

## 7. Custom CNN with SatBlocks

### 7.1 SatBlock building block

Reusable block for deeper feature extraction:

- Two Conv2D layers with the same number of filters.  
- Each convolution followed by BatchNorm and LeakyReLU.  
- MaxPooling2D for downsampling.  
- Dropout with configurable rate.

### 7.2 Architecture

Deeper and more expressive CNN:

- Stem: Conv2D + BatchNorm + LeakyReLU on the raw input.  
- Three SatBlocks in sequence with increasing filters (e.g., 64 → 128 → 256) and tuned dropout.  
- `GlobalAveragePooling2D` instead of Flatten for spatial aggregation.  
- Dense(256) with LeakyReLU and dropout.  
- Final Dense(10) softmax for the 10 EuroSAT classes.

The model summary shows layer names, output shapes, and parameter counts.

### 7.3 Compilation and training

Same setup as the baseline:

- Loss: `sparse_categorical_crossentropy`.  
- Metric: accuracy.  
- Optimizer: Adam with identical hyperparameters.  
- Early stopping on validation loss with best weight restoration.

---

## 8. Evaluation and Visualization

### 8.1 Test‑set performance

For each model (baseline and custom):

- Evaluate on the held‑out test set.  
- Compute overall test accuracy.  
- Generate predictions for all test samples.  
- Produce a classification report (precision, recall, F1 for each class).  
- Build and plot a confusion matrix heatmap.

### 8.2 Training curves

For both CNNs:

- Plot training vs validation loss per epoch.  
- Plot training vs validation accuracy per epoch.

These plots help diagnose underfitting or overfitting and compare learning behavior between the two architectures.

### 8.3 Example predictions (optional)

Optionally:

- Show a grid of test images.  
- Display true and predicted labels for each image.  
- Highlight correct vs incorrect predictions for qualitative inspection.

---

## 9. How to Run

1. Download and extract `EuroSAT_RGB.zip` from the Zenodo EuroSAT record.  
2. Set `original_root` in the notebook to the extracted EuroSAT RGB directory.  
3. Run the notebook in order:
   - Dataset splitting and directory creation.  
   - Data generators and callbacks.  
   - Baseline CNN definition and training.  
   - Custom CNN (with SatBlocks) definition and training.  
   - Evaluation, confusion matrix, and training curves.

Running all cells gives trained models and visual comparisons of baseline vs custom CNN performance on EuroSAT RGB land‑use classification.



