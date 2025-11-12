# Classification Head for Jump-Starting Embedding Separation

## Overview

Added an optional classification head with cross-entropy loss to help jump-start embedding separation during the early phases of training. This approach combines classification loss with triplet loss for the first several epochs, providing strong supervised learning signals that help the model learn discriminative features faster.

## Motivation

Triplet loss alone can be slow to converge initially because:
1. Early embeddings are often randomly distributed
2. Finding meaningful hard triplets is difficult with poor embeddings
3. The margin constraint can be difficult to satisfy initially

By adding a classification head:
- The model learns class-discriminative features immediately
- Embeddings naturally cluster by class from the start
- Triplet mining becomes more effective sooner
- Training converges faster overall

## Implementation Details

### Model Changes (`siamesenet_dino.py`)

1. **Added optional classification head** to `SiameseNet.__init__()`:
   ```python
   self.num_classes = kwargs.get('num_classes', None)
   if self.num_classes is not None and self.num_classes > 0:
       self.classification_head = nn.Sequential(
           nn.Dropout(self.dropout_rate),
           nn.Linear(self.embedding_dim, self.num_classes)
       )
   ```

2. **Updated `forward()` method** to optionally return logits:
   ```python
   def forward(self, x: torch.Tensor, return_logits: bool = False)
   ```
   - When `return_logits=True` and classification head exists, returns `(embeddings, logits)`
   - Otherwise returns just embeddings as before

3. **Updated `forward_triplet()` method** for batch classification:
   - Added `return_logits` parameter
   - Returns embeddings and logits for all three inputs (anchor, positive, negative)

### Training Loop Changes (`training_loop.py`)

1. **Added classification loss function** in `setup_model()`:
   ```python
   if hasattr(self.model, 'classification_head') and self.model.classification_head is not None:
       self.classification_loss_fn = nn.CrossEntropyLoss()
   ```

2. **Updated data loading** to capture player labels:
   ```python
   for i, (anchor, positive, negative, labels) in enumerate(self.dataloader):
   ```

3. **Implemented combined loss strategy**:
   - Uses both triplet loss and classification loss for first N epochs
   - Classification weight decreases linearly from start_weight to 0
   - Formula: `loss = triplet_loss + (class_weight * classification_loss)`
   - After N epochs, reverts to pure triplet loss

4. **Automatic num_classes detection** in `setup_training_pipeline()`:
   - Automatically sets `num_classes` from dataset if enabled
   - Only activates if `training_config.use_classification_head=True`

### Configuration (`all_config.py`)

Added three new parameters to `TrainingConfig`:

```python
use_classification_head: bool = True
classification_epochs: int = 10
classification_weight_start: float = 1.0
```

## Usage

### Default Behavior (Classification Head Enabled)

The classification head is **enabled by default**. When a model is trained:

1. `num_classes` is automatically set from the number of players in the dataset
2. For the first 10 epochs:
   - Both triplet loss and classification loss are computed
   - Classification weight starts at 1.0 and linearly decreases to 0
3. After epoch 10:
   - Only triplet loss is used (standard training)
   - Classification head remains in the model but is not used

### Disabling Classification Head

To disable the classification head feature:

```python
# In config
training_config.use_classification_head = False

# Or when creating a model
model = SiameseNet(num_classes=None)  # Explicitly set to None
```

### Adjusting Classification Training Duration

```python
# Train with classification for 20 epochs instead of 10
training_config.classification_epochs = 20

# Start with higher classification weight
training_config.classification_weight_start = 2.0  # Double the weight
```

## Example Training Flow

**Epoch 1-10** (with classification):
```
Batch forward pass:
  → Get embeddings + logits for (anchor, positive, negative)
  → Compute triplet_loss from embeddings
  → Compute classification_loss from logits vs. player labels
  → Combined loss = triplet_loss + (weight * classification_loss)
  → weight linearly decreases: 1.0 → 0.9 → 0.8 → ... → 0.1 → 0.0
```

**Epoch 11+** (pure triplet):
```
Batch forward pass:
  → Get embeddings only for (anchor, positive, negative)
  → Compute triplet_loss from embeddings
  → loss = triplet_loss
```

## Benefits

1. **Faster convergence**: Models learn discriminative features immediately
2. **Better initialization**: Embeddings start well-separated by class
3. **Improved triplet mining**: Better early embeddings → more effective hard triplet mining
4. **Backward compatible**: Classification head is optional and can be disabled
5. **Flexible**: Duration and weight of classification phase are configurable

## Monitoring

The training loop logs classification loss components periodically:

```
Epoch 1 Batch 10: triplet_loss=0.2847, class_loss=2.1234, class_weight=1.00
Epoch 5 Batch 10: triplet_loss=0.1523, class_loss=0.8921, class_weight=0.50
Epoch 10 Batch 10: triplet_loss=0.0982, class_loss=0.2145, class_weight=0.00
```

Watch for:
- Classification loss should decrease rapidly in early epochs
- Triplet loss should also improve faster than without classification
- After classification phase ends, training should continue smoothly

## Technical Notes

- **Memory**: Classification head adds minimal parameters: `embedding_dim × num_classes`
- **Inference**: Classification head is not used during inference; only embeddings are used
- **Checkpointing**: Classification head state is saved in checkpoints
- **Transfer learning**: A model trained with classification can be fine-tuned without it

## Future Enhancements

Possible improvements:
1. **Adaptive weighting**: Adjust classification weight based on loss values
2. **Hard negative mining with classes**: Use class labels to mine harder negatives
3. **Multi-task learning**: Keep classification loss throughout training with lower weight
4. **Focal loss**: Replace cross-entropy with focal loss for hard example emphasis
