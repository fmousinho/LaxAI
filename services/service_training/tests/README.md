# Service Training Tests

This directory contains organized tests for the service_training module, categorized by test type and scope.

## Test Organization

### 🧩 Unit Tests (`unit/`)

Tests individual components in isolation with minimal dependencies.

**Purpose:**

- Test training loop components with mock/dummy data
- Test utility functions and helpers
- Test model components without external dependencies
- Fast execution, no external service calls

**Examples:**

- `test_training.py` - Training loop with dummy datasets
- `test_checkpoint_naming.py` - Checkpoint naming logic
- `test_model_training_devices.py` - Device handling

**Running Unit Tests:**

```bash
pytest unit/ -v
```

### 🔗 Integration Tests (`integration/`)

Tests components working together, including shared_libs integration.

**Purpose:**

- Test complete training pipelines with shared_libs
- Test WandB logging with shared_libs components
- Test data flow between training components
- May require shared_libs but NOT other services
- Can be slower than unit tests

**Examples:**

- `test_training_suite.py` - Full training pipeline
- `test_wandb_comprehensive.py` - WandB integration
- `test_batch_size_flow.py` - Complete data flow

**Running Integration Tests:**

```bash
pytest integration/ -v
```

### ⚡ Performance Tests (`performance/`)

Tests focused on performance, memory usage, and scalability.

**Purpose:**

- Memory leak detection and monitoring
- Performance benchmarking
- Resource usage analysis
- Long-running tests that may take significant time
- May require special hardware (GPU) or large datasets

**Examples:**

- `test_memory_process_monitoring_improved.py` - Memory monitoring
- `test_wandb_performance_optimized.py` - Performance optimization

**Running Performance Tests:**

```bash
pytest performance/ -v -m performance
```

## Shared Libraries Integration

Tests in the `integration/` directory may use shared_libs components:

- `shared_libs.utils.*` - Utility functions
- `shared_libs.config.*` - Configuration management
- `shared_libs.common.*` - Common components

**Important:** Integration tests should NOT interact with other services (service_cloud, service_tracking, etc.).

## Running All Tests

```bash
# Run all tests
pytest

# Run specific categories
pytest unit/
pytest integration/
pytest performance/

# Run with markers
pytest -m "integration and not slow"
pytest -m performance

# Run with coverage
pytest --cov=src --cov-report=html
```

## Test Markers

- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.skipif` - Conditional test skipping

## Best Practices

1. **Unit Tests:** Use mock objects, avoid external dependencies
2. **Integration Tests:** Test real component interactions, use shared_libs
3. **Performance Tests:** Focus on metrics, may require special setup
4. **No Cross-Service Tests:** Keep service boundaries clear

## Adding New Tests

1. Determine the test category (unit/integration/performance)
2. Place in appropriate directory
3. Use descriptive names: `test_feature_description.py`
4. Add appropriate pytest markers
5. Include docstrings explaining test purpose
