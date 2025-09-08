# TrackGeneratorPipeline Test Suite

This directory contains an independent test suite for the TrackGeneratorPipeline functionality. These tests can be deployed and run independently from the main LaxAI test suite.

## Overview

The TrackGeneratorPipeline is a complex video processing pipeline that handles:
- Video import and validation
- Player detection and tracking
- Crop extraction and upload
- Checkpoint and resume functionality
- Frame-level checkpointing
- URL logging for uploaded crops

## Test Structure

```
tests/track/
├── __init__.py              # Package initialization and common imports
├── conftest.py              # Shared fixtures and pytest configuration
├── test_track_generator_pipeline.py     # Core pipeline functionality tests
├── test_checkpoint_functionality.py     # Checkpoint and resume tests
├── test_url_logging.py       # URL logging functionality tests
└── README.md                 # This documentation
```

## Key Features Tested

### 1. Core Pipeline Functionality
- Pipeline initialization with mocked dependencies
- Run method parameter validation
- Error handling for missing dependencies
- Stop/cancellation functionality
- Result structure validation

### 2. Checkpoint Functionality
- Basic checkpoint save/load operations
- Resume from checkpoint capability
- Frame-level checkpointing (every 100 frames)
- Checkpoint context structure validation
- Resume frame position handling

### 3. URL Logging Functionality
- Crop upload URL formatting (`gs://bucket/path`)
- Batch-level URL logging
- Final summary URL logging
- Cancellation URL logging
- Empty result handling

## Running the Tests

### Prerequisites
- Python 3.8+
- pytest
- The test suite mocks all external dependencies, so no additional setup is required

### Basic Test Execution
```bash
# Run all tests
pytest tests/track/

# Run specific test file
pytest tests/track/test_track_generator_pipeline.py

# Run with verbose output
pytest tests/track/ -v

# Run specific test
pytest tests/track/test_checkpoint_functionality.py::TestCheckpointFunctionality::test_checkpoint_save_method_exists -v
```

### Test Markers
```bash
# Run only checkpoint-related tests
pytest tests/track/ -m checkpoint

# Run only URL logging tests
pytest tests/track/ -m url_logging

# Skip slow tests
pytest tests/track/ -m "not slow"
```

## Mock Strategy

The test suite uses comprehensive mocking to isolate the TrackGeneratorPipeline:

### Mocked Dependencies
- **OpenCV (cv2)**: Video capture and image processing
- **Supervision**: Detection object handling
- **PyTorch/TorchVision**: ML model operations
- **PIL**: Image processing
- **Google Cloud Storage**: File upload/download operations
- **Internal Modules**: Detection models, trackers, path managers

### Mock Configuration
- All mocks are configured in `conftest.py` for global availability
- Fixtures provide consistent mock objects across tests
- Mock behavior can be customized per test as needed

## Test Categories

### Unit Tests
- Individual method testing
- Parameter validation
- Error condition handling
- Mock interaction verification

### Integration Tests
- End-to-end pipeline flow testing
- Component interaction validation
- Checkpoint save/load cycles
- URL logging workflow verification

### Functional Tests
- Real-world usage scenario simulation
- Performance characteristic validation
- Edge case handling

## Test Data

The test suite includes:
- Mock video files and metadata
- Sample detection results
- Checkpoint data structures
- Upload task configurations
- Error scenarios and edge cases

## Continuous Integration

These tests are designed to run in CI/CD environments:

```yaml
# Example GitHub Actions configuration
- name: Run Track Pipeline Tests
  run: |
    cd tests/track
    pytest . -v --tb=short --cov=track_generator_pipeline --cov-report=xml
```

## Extending the Test Suite

### Adding New Tests
1. Create new test methods in existing files or new files
2. Use existing fixtures from `conftest.py`
3. Follow the naming convention: `test_<feature>_<scenario>`
4. Add appropriate pytest markers

### Adding New Fixtures
1. Add fixtures to `conftest.py` for global availability
2. Or add to specific test files for local use
3. Document fixture purpose and usage

### Mock Customization
1. Extend existing mocks in `conftest.py`
2. Add test-specific mock behavior using `patch` decorators
3. Ensure mocks don't interfere with other tests

## Test Coverage

The test suite aims to cover:
- ✅ Pipeline initialization and configuration
- ✅ Video processing workflow
- ✅ Detection and tracking operations
- ✅ Crop extraction and upload
- ✅ Checkpoint save/load operations
- ✅ Frame-level resume functionality
- ✅ URL logging and reporting
- ✅ Error handling and edge cases
- ✅ Cancellation and interruption handling

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure `PYTHONPATH` includes the `src` directory
2. **Mock Conflicts**: Check that mocks are properly isolated per test
3. **Fixture Issues**: Verify fixture dependencies and scope

### Debug Mode
```bash
# Run with debug output
pytest tests/track/ -v -s --pdb

# Show fixture information
pytest tests/track/ --fixtures
```

## Deployment

This test suite can be deployed independently:

```bash
# Copy the test directory
cp -r tests/track /path/to/deployment/tests/

# Install dependencies (minimal)
pip install pytest mock

# Run tests
cd /path/to/deployment/tests/track
pytest .
```

The test suite is self-contained and requires only the TrackGeneratorPipeline source code and basic testing dependencies.
