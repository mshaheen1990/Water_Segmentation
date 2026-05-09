# AGENTS.md

## Project

This project trains water-mask segmentation models using Planet imagery, HEC-RAS rasters, and QGIS-derived water masks.

## Important files

The data manifest and UUID split were already created and verified in Google Colab.

Use these fixed files:
- clean_manifest.csv
- uuid_split_seed42.json

Do not regenerate them unless explicitly asked.

## Data access

The data are stored in Google Drive and are not included in this repository.

The code should load paths from:

configs/paths_colab.yaml

Example paths are shown in:

configs/paths_colab.yaml.example

## Required manifest columns

The clean manifest contains:

- UUID
- date
- planet_path
- hecras_path
- qmask_path
- planet_path_exists
- hecras_path_exists
- qmask_path_exists

## Evaluation rules

- Do not change clean_manifest.csv.
- Do not regenerate uuid_split_seed42.json.
- Do not change the held-out test UUIDs.
- Do not use the held-out test set for model selection.
- Use GroupKFold only on development UUIDs.
- Keep 64x64 center-crop evaluation separate from full-image tiled evaluation.
- Log every experiment.
- First reproduce U-Net early fusion before trying improvements.

## Current baseline reference

Current Colab manifest-based U-Net early-fusion result:

- Input: Planet + HEC-RAS
- Image size: 64
- Threshold: 0.5
- Held-out samples: 336
- Mean IoU: 0.817698
- Mean Dice: 0.892045

Earlier notebook result was approximately:
- Mean IoU: 0.833
- Mean Dice: 0.904

First priority is to make the original notebooks read the new clean manifest and reproduce the baseline as closely as possible.
