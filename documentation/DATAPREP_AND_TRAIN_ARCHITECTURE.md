
# Dataprep and Train Architecture


This document describes the folder structure, data flow, and architectural responsibilities of the `common` and `train` modules in the LaxAI project. It explains how raw data is ingested, processed, and used for model training and evaluation, with a focus on reproducibility and modularity.

## Overview

The LaxAI project is designed for modular, scalable machine learning pipelines focused on lacrosse player re-identification. The codebase is divided into logical modules, each with clear responsibilities and well-defined data storage conventions. Data is managed both locally and in Google Cloud Storage (GCS) to support collaboration, reproducibility, and scalability.

## Folder Structure

### 1. `core/common/`
- **Purpose:** Contains shared utilities, pipeline steps, Google Storage helpers, and common data processing logic. This module abstracts away low-level details and provides robust, reusable components for the rest of the project.
- **Key Folders/Files:**
  - `background_mask.py`, `crop_utils.py`, `detection_utils.py`: Image and video processing utilities for extracting, masking, and cropping frames.
  - `google_storage.py`: Handles all Google Cloud Storage (GCS) interactions, including authentication, downloading/uploading datasets, models, logs, and managing multi-tenant storage.
  - `pipeline.py`, `pipeline_step.py`: Base classes for building modular, step-based pipelines with status tracking and error handling.
- **Data Usage:**
  - Reads and writes data to GCS buckets (e.g., raw videos, processed frames, model checkpoints, evaluation reports).
  - Provides interfaces for other modules to access shared resources and data, ensuring consistency and security.
  - Supports both local and cloud-based workflows, making it easy to switch between development and production environments.

### 2. `core/train/`
- **Purpose:** Implements the training pipeline, dataset preparation, augmentation, and evaluation logic. This module is responsible for transforming raw data into high-quality datasets, training deep learning models, and evaluating their performance.
- **Key Folders/Files:**
  - `dataprep_pipeline.py`: Downloads raw videos from GCS, extracts frames, applies augmentations, and prepares datasets for training. Handles data validation, splitting, and upload of processed data.
  - `dataset.py`: Defines the `LacrossePlayerDataset` class for loading and validating player image crops, supporting advanced filtering and augmentation.
  - `train_pipeline.py`: Orchestrates the end-to-end training process, including dataset creation, model training, and evaluation. Integrates with pipeline steps for modularity and error handling.
  - `augmentation.py`: Contains image augmentation logic for robust model training, supporting both offline and online augmentation strategies.
- **Data Usage:**
  - Reads raw videos and metadata from GCS (via `common` utilities), ensuring all data is up-to-date and accessible.
  - Writes processed datasets (player crops, train/val splits) to local or cloud storage for reproducibility and sharing.
  - Saves trained models, checkpoints, and evaluation results to GCS for later use, enabling model versioning and collaborative analysis.
  - Optionally logs intermediate results and statistics for debugging and monitoring.

### 3. `config/`
- **Purpose:** Centralizes configuration for all modules, ensuring that paths, bucket names, and transformation settings are consistent and easy to update.
- **Key Files:**
  - `all_config.py`, `transforms_config.py`, `store_config.py`: Define paths, bucket names, transformation settings, and other global parameters.
- **Data Usage:**
  - Provides configuration values (e.g., GCS bucket names, local paths, augmentation modes) to both `common` and `train` modules.
  - Enables easy switching between development, testing, and production environments by changing a single config file.

### 4. Data Flow Example
1. **Raw Data Ingestion:**
   - Raw videos and metadata are uploaded to a GCS bucket (e.g., `laxai_dev`).
   - The `common` module provides secure, authenticated access to these resources for downstream processing.
2. **Data Preparation:**
   - `dataprep_pipeline.py` downloads videos, extracts frames, and creates player crop datasets, applying augmentations and validation checks.
   - Processed datasets are saved locally for fast access and/or uploaded back to GCS for sharing and backup.
   - Data splits (train/val/test) and metadata are generated and stored alongside the datasets.
3. **Training:**
   - `train_pipeline.py` loads the prepared dataset, trains the model using configurable parameters, and saves checkpoints at regular intervals.
   - Trained models, logs, and training metadata are uploaded to GCS for reproducibility, versioning, and sharing with collaborators.
4. **Evaluation:**
   - Evaluation results, reports, and visualizations are stored in GCS or local folders for further analysis and comparison.
   - The evaluation pipeline supports cross-validation, ranking metrics, and detailed reporting for model selection.

## Best Practices
- All data movement between local and cloud storage is handled via the `core.common.google_storage` utilities, which support multi-tenant and secure access patterns.
- Configuration files in `config/` should be updated to reflect any changes in storage paths, bucket names, or transformation settings.
- Intermediate and final results (datasets, models, logs, reports) should be saved to GCS to ensure reproducibility, backup, and collaboration.
- Use modular pipeline steps and context dictionaries to enable flexible, debuggable, and restartable workflows.
- Document any changes to the data structure or pipeline logic in the appropriate module docstrings and this architecture document.

---

For more details, see the docstrings in each module and the configuration files in the `config/` folder. For advanced usage, refer to the example scripts and pipeline step documentation.
