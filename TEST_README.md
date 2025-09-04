# Test Structure and Execution

This document describes the test structure and execution order for the LaxAI project.

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Examples**:
  - `test_dataloader_restart_unit.py` - DataLoader restart functionality
  - `test_timebox_cancellation_unit.py` - Timebox cancellation mechanism
- **Execution**: Fast, no external dependencies

### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions and external service integration
- **Examples**:
  - `test_training_suite.py` - Training pipeline integration
  - `test_training.py` - Training components integration
- **Execution**: May require external services, moderate speed

### Performance Tests (`tests/performance/`)
- **Purpose**: Test memory usage, timing, and performance characteristics
- **Examples**:
  - `test_memory_leak_fixes.py` - Memory leak detection
  - `test_memory_management.py` - Memory management validation
- **Execution**: Slower, resource-intensive

### End-to-End Tests (`tests/integration/` with `@pytest.mark.e2e`)
- **Purpose**: Full pipeline testing with real data and external services
- **Examples**:
  - `test_train_all_resnet_with_two_datasets_memory_stable` - ResNet training (2 epochs, 2 datasets)
  - `test_train_all_with_dino_memory_stable` - DINO training (1 epoch, 1 dataset)
- **Execution**: Slowest, most comprehensive, run last

## Test Execution Order

Tests are designed to run in the following order to ensure stability and efficiency:

1. **Unit Tests** (`tests/unit/`) - Fastest, isolated component testing
2. **API Tests** (`tests/api/`) - API endpoint and service testing
3. **Integration Tests** (`tests/integration/`) - Component interaction testing (excluding e2e)
4. **Performance Tests** (`tests/performance/`) - Resource usage and performance validation
5. **End-to-End Tests** (`tests/integration/` with `@pytest.mark.e2e`) - Full pipeline validation (slowest, run last)

## Running Tests

### Option 1: Automated Test Runner (Recommended)
```bash
./run_tests.py
```
This script automatically runs tests in the correct order: **unit → api → integration → performance → e2e**.

### Option 2: Manual Execution

Run specific test categories in order:
```bash
# 1. Unit tests
python -m pytest tests/unit/ -v

# 2. API tests
python -m pytest tests/api/ -v

# 3. Integration tests (excluding e2e)
python -m pytest tests/integration/ -m "not e2e" -v

# 4. Performance tests
python -m pytest tests/performance/ -v

# 5. End-to-end tests (run last)
python -m pytest -m "e2e" -v
```

## Test Markers

- `@pytest.mark.e2e` - End-to-end tests (run last)
- `@pytest.mark.slow` - Tests that take > 30 seconds
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.fast` - Tests that complete < 30 seconds

## Memory Stability Assertions

End-to-end tests include memory stability assertions:

- **ResNet Test**: Memory delta ≤ 200MB, leak detection ≤ 50MB
- **DINO Test**: Memory delta ≤ 150MB, leak detection ≤ 30MB (accounts for model downloads)

## Environment Requirements

- Python 3.12+
- Virtual environment: `.venv31211`
- Required packages in `requirements-dev.txt`
- Environment secrets configured via `utils/env_secrets.py`

## Troubleshooting

### Common Issues

1. **Memory Issues**: Ensure sufficient RAM (> 8GB recommended for e2e tests)
2. **Timeout Errors**: E2E tests may take 5-15 minutes each
3. **External Service Failures**: Ensure GCS, WandB, and HuggingFace access
4. **Environment Secrets**: Run `python -c "from utils.env_secrets import setup_environment_secrets; setup_environment_secrets()"`
