# Data Storage Organization

This document describes the organization of data directories and storage conventions used in Google Cloud Storage (GCS) for the LaxAI project. It covers the structure and purpose of the main directories, including `common`, `user`, `train`, and those related to video files and processed data.

## Overview

LaxAI uses Google Cloud Storage to manage all large-scale data, including raw videos, processed datasets, model checkpoints, logs, and evaluation results. The storage structure is designed to support multi-tenant workflows, reproducibility, and efficient collaboration.

## Main GCS Directories and Conventions

The LaxAI project organizes all data in a single bucket (e.g., `laxai_dev`) using a `user_path` prefix for each client or tenant. The structure is as follows:

### 1. Tenant Data (`<tenant_id>/user/`)
- **Purpose:** Stores all tenant-specific data, including raw videos, processed runs, crops, detections, datasets, and metadata.
- **Structure:**
  - `<tenant_id>/user/raw/` — Raw uploaded videos for the tenant. Videos are initially uploaded here and are not yet associated with a processing run.
  - `<tenant_id>/user/<run_folder>/video_<guid>/` — All data for a specific processing run:
    - `video_<guid>.mp4` — The original video file, moved from `raw/` and renamed with a GUID for traceability. This move is performed by the pipeline at the start of processing.
    - `metadata.json` — Metadata for the video
    - `detections.json` — Detection results for the video
    - `selected_frames/` — Selected frames for the run
    - `crops/original/` — Original player crops
    - `crops/modified/` — Background-removed player crops
    - `datasets/` — Contains dataset splits and files for training/validation (e.g., `train/`, `val/` folders with images and labels)
    - (other subfolders as needed for pipeline steps)
- **Usage:**
  - All pipeline steps read and write to these subfolders. Each run is isolated in its own folder for reproducibility and traceability.
  - The original video is always moved from `raw/` to a new GUID-named directory at the start of a run, ensuring that the raw upload area only contains unprocessed files and that all processed data is grouped by run.

### 2. Shared Resources (`common/`)
- **Purpose:** Stores shared models and reference data accessible to all tenants and pipelines.
- **Structure:**
  - `common/models/` — Pretrained and shared models
  - `common/` — Other shared resources (e.g., reference datasets, config files)
- **Usage:** Used for model initialization, evaluation, and as a source of baseline resources.

## Example GCS Structure

```
laxai_dev/
  common/
    models/
    reference_datasets/
    config.yaml
  tenant1/
    user/
      raw/
        game1.mp4
        game2.mp4
      run_2025_07_24_001/
        video_abc12345/
          video_abc12345.mp4
          metadata.json
          detections.json
          selected_frames/
            frame_001.jpg
            ...
          crops/
            original/
              frame_001/...
            modified/
              frame_001/...
          datasets/
            train/
              frame_001/
                crop_001/
                  crop.jpg
                  crop_aug1.jpg
                  ...
                crop_002/
                  crop.jpg
                  ...
                ...
              frame_002/
                crop_003/
                  crop.jpg
                  ...
                ...
            val/
              frame_001/
                crop_004/
                  crop.jpg
                  ...
                ...
              ...
          #
          # Each frame directory (e.g., frame_001/) inside crops/original/, crops/modified/, and datasets/train/ or datasets/val/ contains subdirectories for each crop (e.g., crop_<tracker_id>/crop.jpg, crop_aug1.jpg, etc.).
          # This includes both original crops and those created via augmentation (e.g., augmented versions of the same crop with different transformations).
          # Augmented crops are typically named or organized to indicate their origin or transformation, but the exact naming is determined by the augmentation logic in the pipeline.
      run_2025_07_25_001/
        ...
  tenant2/
    user/
      raw/
      run_2025_07_24_001/
        ...
```

## Best Practices
- Use the `common/` directory for all resources that should be accessible to every user or pipeline.
- Store all tenant-specific data under `<tenant_id>/user/`, with each run in its own subfolder for reproducibility.
- Keep raw videos in `<tenant_id>/user/raw/` and processed data in run-specific folders.
- Use consistent naming conventions for runs, videos, and metadata files to ensure traceability.
- Always use the Google Storage client (`get_storage(user_path)`) to ensure correct path handling and permissions.

---

For more details on how these directories are used in code, see the `core.common.google_storage` module and the configuration files in the `config/` folder.
