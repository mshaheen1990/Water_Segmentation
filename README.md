# Water_Segmentation

# Water-Mask Segmentation Using Planet Imagery and HEC-RAS Data

This project trains and evaluates deep-learning models for binary water-mask segmentation using Planet satellite imagery, HEC-RAS hydraulic-model rasters, and QGIS-derived water-mask labels.

The main goal is to compare Planet-only segmentation models against early-fusion models that combine Planet imagery with HEC-RAS data.

## Project Status

This project is currently being prepared for controlled and reproducible experiment runs.

The data-preparation step has already been completed in Google Colab. A clean manifest file and a locked UUID-based split file were created and verified.

## Dataset Setup

The dataset uses three data sources:

1. Planet imagery
2. HEC-RAS raster data
3. QGIS-derived water-mask labels

The original file paths were stored in three Excel files:

- Planet image paths
- HEC-RAS raster paths
- QGIS mask paths

These Excel files were merged into a single clean manifest.

## Clean Manifest

The training pipeline should use the following fixed manifest file:

```text
/content/drive/MyDrive/WaterMaskProject/results/clean_manifest.csv

The clean manifest contains the following columns:

UUID
date
planet_path
hecras_path
qmask_path
planet_path_exists
hecras_path_exists
qmask_path_exists

The manifest contains 2,076 valid samples. All Planet, HEC-RAS, and QGIS mask paths were verified to exist.

Locked UUID Split

The train/development and held-out test split is stored in:

/content/drive/MyDrive/WaterMaskProject/splits/uuid_split_seed42.json

This split must not be regenerated unless explicitly required.

The split was created by UUID to avoid leakage between training, validation, and testing.

Current split:

Development set: 48 UUIDs
Held-out test set: 9 UUIDs
Total UUIDs: 57
Total samples: 2,076
Evaluation Rules

All experiments must follow these rules:

Do not modify clean_manifest.csv.
Do not regenerate uuid_split_seed42.json.
Do not use held-out test UUIDs for training, validation, threshold tuning, model selection, or hyperparameter tuning.
Use GroupKFold cross-validation only on the development UUIDs.
Use the held-out test set only once for final evaluation after model selection.
Keep 64 × 64 center-crop evaluation separate from full-image tiled evaluation.
Log every experiment.
Current Baseline

The current main baseline is:

Model: U-Net
Input: Planet + HEC-RAS early fusion
Image size: 64 × 64
Planet channels: 8
HEC-RAS channels: 1
Total channels: 9
Loss: Binary cross-entropy + Dice loss
Optimizer: Adam
Learning rate: 1e-4
Batch size: 16
Prediction threshold: 0.5

Previous held-out test performance was approximately:

Mean IoU: 0.833
Mean Dice: 0.904
Models to Compare

The main models are:

U-Net with Planet-only input
U-Net with Planet + HEC-RAS early fusion
ResUNet with Planet-only input
ResUNet with Planet + HEC-RAS early fusion
TFN U-Net with Planet + HEC-RAS
Optional compression or explainability variants
Recommended Workflow
Load clean_manifest.csv.
Load uuid_split_seed42.json.
Split the manifest into development and held-out test sets using UUID.
Run 5-fold GroupKFold cross-validation on the development set.
Select model settings using validation performance only.
Train the final model on the full development set.
Evaluate once on the locked held-out test set.
Save fold-level results and final test results.
Google Colab Paths

The project is designed to run in Google Colab with Google Drive mounted at:

/content/drive

A local path configuration file should be created as:

configs/paths_colab.yaml

Example:

project_root: /content/drive/MyDrive/WaterMaskProject

clean_manifest: /content/drive/MyDrive/WaterMaskProject/results/clean_manifest.csv
split_file: /content/drive/MyDrive/WaterMaskProject/splits/uuid_split_seed42.json

results_dir: /content/drive/MyDrive/WaterMaskProject/results
checkpoint_dir: /content/drive/MyDrive/WaterMaskProject/checkpoints

Do not commit private data files or large TIFF imagery to GitHub.

Notes for Codex

When using Codex:

Use the existing notebooks as the source of truth.
Do not rewrite the project from scratch unless asked.
Refactor the code into reusable scripts only after preserving the current behavior.
Do not regenerate the manifest or UUID split.
First reproduce the U-Net early-fusion baseline.
Only after baseline reproduction, create controlled improvement experiments.
All improvements must be selected using GroupKFold validation, not held-out test performance.
