# Evaluation Data Usage Summary

## What Data is Used for Model Evaluation?

The evaluation system now properly uses your existing train/validation split instead of creating random splits.

### Data Source Structure

Your dataset should be organized as:
```
dataset_path/
├── train/           # Training data (used for model training and cross-validation)
│   ├── player_001/
│   ├── player_002/
│   └── ...
└── val/             # Validation data (PRIMARY EVALUATION DATA)
    ├── player_001/
    ├── player_002/
    └── ...
```

### Evaluation Data Usage

#### 1. **Primary Evaluation Metrics** 
- **Data Used**: `val/` folder contents
- **Purpose**: Main evaluation metrics (Rank-1, Rank-5, mAP, Accuracy, etc.)
- **Player Split**: Uses the players that were allocated to validation during dataset creation

#### 2. **Cross-Validation**
- **Data Used**: `train/` folder contents  
- **Purpose**: Robustness assessment with K-fold validation
- **Player Split**: Splits training players into K folds

#### 3. **Distance Analysis**
- **Data Used**: `val/` folder contents
- **Purpose**: Embedding quality assessment (same vs different player distances)

### Key Benefits

✅ **No Data Leakage**: Respects your existing train/val split
✅ **Proper Evaluation**: Uses held-out validation data for true performance assessment
✅ **Consistent Results**: Same validation set used across all evaluation runs
✅ **Player-Level Split**: Maintains player separation between train and validation

### Data Flow in Pipeline

1. **Training Phase**: Model trained on `train/` folder
2. **Evaluation Phase**: Model evaluated on `val/` folder
3. **Cross-Validation**: Additional robustness check using `train/` folder with K-fold splits

### Example Evaluation Output

When evaluation runs, you'll see logs like:
```
Using existing validation split:
  Train dir: /path/to/dataset/train
  Val dir: /path/to/dataset/val
Generating embeddings for validation set...
Created subset with 5 players, 250 images
```

This confirms that:
- 5 players were in your validation split
- 250 total validation images are being used for evaluation
- The model is being tested on completely separate data from training

### Transforms Used

- **Training Data**: Uses 'training' transforms (with augmentation)
- **Validation Data**: Uses 'validation' transforms (typically without augmentation for consistent evaluation)

This ensures fair evaluation by using appropriate transforms for each data split.
