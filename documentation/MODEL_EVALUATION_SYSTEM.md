# Model Evaluation System Documentation

## Overview

The `ModelEvaluator` class provides comprehensive evaluation for Siamese networks used in lacrosse player re-identification. It implements multiple evaluation methodologies following best practices for deep metric learning and person re-identification tasks.

## Evaluation Methodologies

### 1. Distance-Based Metrics

These metrics analyze the separability of embeddings in the feature space:

- **Average Distance (Same Player)**: Mean Euclidean distance between embeddings of the same player
- **Average Distance (Different Players)**: Mean Euclidean distance between embeddings of different players  
- **Distance Separation**: Difference between inter-class and intra-class distances (higher is better)
- **Cosine Similarity Metrics**: Similar analysis using cosine similarity instead of Euclidean distance

**Purpose**: Validates that the model learns meaningful embeddings where same-player images cluster together and different-player images are well-separated.

### 2. Classification Metrics

Binary classification performance treating similarity as a threshold-based decision:

- **Accuracy**: Overall percentage of correct same/different predictions
- **Precision**: Of all "same player" predictions, how many were correct
- **Recall**: Of all actual same-player pairs, how many were correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

**Purpose**: Evaluates the model's ability to distinguish between same and different players using a similarity threshold.

### 3. Ranking Metrics (Most Important for Re-ID)

These metrics evaluate how well the model ranks candidate matches:

- **Rank-1 Accuracy**: Percentage of queries where the correct match is the top result
- **Rank-5 Accuracy**: Percentage of queries where the correct match is in the top 5 results
- **Mean Average Precision (mAP)**: Average precision across all query-gallery comparisons
- **Cumulative Matching Characteristic (CMC)**: Performance across different rank thresholds

**Purpose**: These are the gold standard metrics for person re-identification systems, measuring real-world retrieval performance.

### 4. Cross-Validation

K-fold cross-validation ensuring player-level splits:

- **Player-Based Splitting**: Ensures no player appears in both training and validation sets
- **Multiple Metrics**: Computes mean and standard deviation for accuracy, F1-score, and rank-1 accuracy
- **Robustness Assessment**: Validates model generalization across different player combinations

**Purpose**: Provides robust performance estimates and detects overfitting to specific players.

## Usage Examples

### Basic Evaluation

```python
from core.train.evaluator import ModelEvaluator
from core.train.siamesenet import SiameseNet

# Initialize evaluator
model = SiameseNet(embedding_dim=128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
evaluator = ModelEvaluator(model=model, device=device)

# Run comprehensive evaluation
results = evaluator.evaluate_comprehensive(dataset, test_split=0.2)

# Generate report
report = evaluator.generate_evaluation_report(results)
print(report)
```

### Integration with Training Pipeline

```python
from core.train.train_pipeline import TrainPipeline

# Initialize pipeline
pipeline = TrainPipeline(tenant_id="tenant1", verbose=True)

# Run complete pipeline with evaluation
context = {"dataset_path": "/path/to/dataset"}
context = pipeline._create_dataset(context)
context = pipeline._train_model(context)
context = pipeline._evaluate_model(context)

# Access results
evaluation_summary = context['evaluation_summary']
```

## Key Performance Indicators

For lacrosse player re-identification, focus on these metrics:

1. **Rank-1 Accuracy > 0.80**: At least 80% of queries should return the correct player as the top match
2. **Rank-5 Accuracy > 0.95**: 95% of queries should have the correct player in top 5 matches
3. **Mean Average Precision > 0.70**: Good overall retrieval quality
4. **Distance Separation > 0.5**: Clear separation between same/different player embeddings
5. **Cross-Validation Std < 0.05**: Consistent performance across different data splits

## Best Practices

## Data Usage for Evaluation

The evaluation system is designed to work with your existing train/validation split:

### Expected Dataset Structure
```
dataset_path/
├── train/
│   ├── player1/
│   │   ├── crop_001.jpg
│   │   ├── crop_002.jpg
│   │   └── ...
│   ├── player2/
│   └── ...
└── val/
    ├── player1/
    │   ├── crop_100.jpg
    │   ├── crop_101.jpg
    │   └── ...
    ├── player2/
    └── ...
```

### Data Usage Strategy

1. **Training Data** (`train/` folder): Used for cross-validation and model training
2. **Validation Data** (`val/` folder): Used for primary evaluation metrics
3. **No Data Leakage**: The system respects your existing train/val split
4. **Player-Level Split**: Evaluation ensures no player appears in both train and val

### Fallback Behavior
If the `val/` directory is not found, the system will:
- Log a warning
- Fall back to creating a random 80/20 split from the training data
- Ensure player-level splitting (no player in both sets)

### Threshold Selection
- **Validation-Based**: Use validation set to find optimal similarity threshold
- **ROC Analysis**: Consider multiple thresholds and select based on precision-recall trade-offs
- **Task-Specific**: Adjust threshold based on whether false positives or false negatives are more costly

### Evaluation Frequency
- **During Training**: Monitor rank-1 accuracy every few epochs
- **Model Selection**: Use validation mAP for model selection
- **Final Assessment**: Run full evaluation suite on held-out test set

## Weights & Biases Integration

The evaluator automatically logs results to W&B when enabled:

```python
# Metrics logged under 'eval/' prefix:
eval/classification/accuracy
eval/ranking/rank_1_accuracy
eval/ranking/mean_average_precision
eval/distance/distance_separation
eval/cv/accuracy_mean
eval/cv/accuracy_std
```

## Output Files

The evaluator saves detailed results:

- `evaluation_results/evaluation_results.json`: Complete results in JSON format
- Console output: Human-readable summary report
- W&B dashboard: Interactive metrics and visualizations

## Common Issues and Solutions

### Low Rank-1 Accuracy
- Check distance separation - embeddings may not be well-separated
- Increase model capacity or training epochs
- Verify data quality and labeling accuracy

### High Variance in Cross-Validation
- May indicate overfitting to specific players
- Consider data augmentation or regularization
- Ensure sufficient training data per player

### Poor Distance Separation
- Model may not be learning discriminative features
- Check loss function and training convergence
- Consider different embedding dimensions or architectures

## Extending the Evaluator

To add custom evaluation metrics:

```python
class CustomEvaluator(ModelEvaluator):
    def _evaluate_custom_metric(self, embeddings, labels):
        # Implement custom evaluation logic
        return {"custom_metric": value}
    
    def evaluate_comprehensive(self, dataset, test_split=0.2):
        results = super().evaluate_comprehensive(dataset, test_split)
        results['custom_metrics'] = self._evaluate_custom_metric(
            embeddings, labels
        )
        return results
```

This evaluation system provides a comprehensive assessment of model performance using industry-standard metrics for person re-identification tasks.
