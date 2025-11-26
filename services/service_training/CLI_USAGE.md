# Training CLI - JSON Parameter Usage Guide

## Overview
The training CLI has been refactored to use Pydantic schema-based validation with JSON parameters. This provides:
- ✅ **Type safety** through Pydantic schema validation
- ✅ **Consistency** with the HTTP API interface
- ✅ **Flexibility** to pass any combination of parameters
- ✅ **Default values** from configuration when params are omitted

## Basic Usage

### Simple run with defaults
```bash
python -m cli.train_cli --tenant-id tenant1 --custom-name my_training_run
```

### With custom training parameters
```bash
python -m cli.train_cli --tenant-id tenant1 \
    --custom-name experiment_001 \
    --training-params '{"num_epochs": 100, "batch_size": 32, "lr_initial": 0.001}'
```

### With model and eval parameters
```bash
python -m cli.train_cli --tenant-id tenant1 \
    --training-params '{"num_epochs": 50, "batch_size": 64}' \
    --model-params '{"embedding_dim": 512, "dropout_rate": 0.1}' \
    --eval-params '{"batch_size": 128, "number_of_workers": 4}'
```

## Available Parameters

### Training Parameters (`--training-params`)
```json
{
  "num_epochs": 100,
  "batch_size": 128,
  "num_workers": 0,
  "prefetch_factor": 4,
  "lr_initial": 0.001,
  "lr_scheduler_patience": 5,
  "lr_scheduler_factor": 0.5,
  "early_stopping_patience": 15,
  "dataset_address": "gs://bucket/path/to/dataset",
  "margin": 0.3,
  "weights": "checkpoint"  // Options: "checkpoint", "latest", "reset"
}
```

### Model Parameters (`--model-params`)
```json
{
  "embedding_dim": 512,
  "dropout_rate": 0.05,
  "input_height": 224,
  "input_width": 224,
  "model_class_module": "siamesenet_dino",
  "model_class": "SiameseNet"
}
```

### Eval Parameters (`--eval-params`)
```json
{
  "number_of_workers": 0,
  "batch_size": 64,
  "prefetch_factor": 2
}
```

## Workflow-specific Arguments

These are still available as regular CLI flags:

```bash
--tenant-id TENANT_ID          # Required: Tenant identifier
--custom-name CUSTOM_NAME      # Custom name for the run
--verbose                      # Enable verbose logging
--resume-from-checkpoint       # Resume from checkpoint (default: True)
--wandb-tags TAG1 TAG2         # WandB tags (space-separated)
--task-id TASK_ID              # Task ID for tracking
--auto-resume-count N          # Auto-resume attempt count
```

## Examples

### Quick experiment with different learning rate
```bash
python -m cli.train_cli \
    --tenant-id prod_tenant \
    --custom-name "lr_experiment_0001" \
    --training-params '{"lr_initial": 0.0005, "num_epochs": 50}'
```

### Full configuration override
```bash
python -m cli.train_cli \
    --tenant-id prod_tenant \
    --custom-name "full_config_test" \
    --training-params '{
        "num_epochs": 200,
        "batch_size": 64,
        "lr_initial": 0.0001,
        "early_stopping_patience": 20,
        "weights": "latest"
    }' \
    --model-params '{
        "embedding_dim": 1024,
        "dropout_rate": 0.2
    }' \
    --eval-params '{
        "batch_size": 128
    }' \
    --wandb-tags experiment baseline
```

### Using specific dataset
```bash
python -m cli.train_cli \
    --tenant-id tenant1 \
    --custom-name "dataset_specific_run" \
    --training-params '{"dataset_address": "gs://my-bucket/tenant1/datasets/dataset_001"}'
```

## Migration from Old CLI

### Old CLI (deprecated)
```bash
python -m cli.train_cli \
    --tenant-id tenant1 \
    --num-epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

### New CLI (current)
```bash
python -m cli.train_cli \
    --tenant-id tenant1 \
    --training-params '{"num_epochs": 100, "batch_size": 32, "lr_initial": 0.001}'
```

## Error Handling

The CLI will validate JSON syntax and schema compliance before starting training:

```bash
# Invalid JSON
$ python -m cli.train_cli --training-params '{invalid json}'
❌ Error parsing --training-params: Invalid JSON
   Expecting property name enclosed in double quotes: line 1 column 2 (char 1)

# Invalid schema field
$ python -m cli.train_cli --training-params '{"invalid_field": 123}'
❌ Error validating --training-params against schema
   1 validation error for TrainingParams
   invalid_field
     Extra inputs are not permitted [type=extra_forbidden, ...]
```

## Tips

1. **Use single quotes** around JSON strings in bash to avoid escaping issues
2. **Omit parameters** you don't want to change - defaults from config will be used
3. **Check schema** in `schemas/training.py` for available fields
4. **Field names** may differ from old CLI args (e.g., `lr_initial` not `learning_rate`)
