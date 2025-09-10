# 🚀 LaxAI Multi-Service Development Guide

This workspace is organized as a multi-service architecture to improve development workflow, testing, and deployment strategies.

## 📁 Service Organization

### 🎯 **Tracking Service** (`src/track/`)

- **Purpose**: Video processing and player tracking pipeline
- **Key Files**:
  - `unverified_track_generator_pipeline.py` - Main pipeline
  - Tests: `tests/track/`
- **Debug Configs**:
  - `🎯 Track: Debug Pipeline` - Debug the tracking pipeline
  - `🎯 Track: Run Tests` - Run tracking tests

### 🧠 **Training Service** (`src/train/`)

- **Purpose**: Model training and fine-tuning
- **Key Files**:
  - `scripts/train_workflow.py` - Main training script
  - Tests: `tests/train/`
- **Debug Configs**:
  - `🧠 Train: Debug Training Script` - Debug training process
  - `🧠 Train: Run Tests` - Run training tests

### 🌐 **API Service** (`src/api/`)

- **Purpose**: REST API endpoints and web interface
- **Key Files**:
  - `v1/main.py` - API server entry point
  - Tests: `tests/api/`
- **Debug Configs**:
  - `🌐 API: Debug Server` - Debug the API server
  - `🌐 API: Run Tests` - Run API tests

### ☁️ **Cloud Service** (`src/cloud/`)

- **Purpose**: Cloud and batch processing functions
- **Key Files**:
  - `batch_processor.py` - Batch processing logic
- **Debug Configs**:
  - `☁️ Cloud: Debug Batch Processing` - Debug cloud processing

### 🔧 **Shared Components**

- **Common** (`src/common/`) - Shared utilities and classes
- **Config** (`src/config/`) - Configuration management
- **Utils** (`src/utils/`) - General utilities

## 🎯 Quick Start Guide

### Running Services

1. **Debug a Service**: Use `Ctrl+Shift+D` (or `Cmd+Shift+D` on Mac) and select the appropriate debug configuration
2. **Run Tasks**: Use `Ctrl+Shift+P` → "Tasks: Run Task" and select a service-specific task
3. **Run Tests**: Use the Test Explorer or debug configurations for service-specific tests

### Common Workflows

#### 🎯 Working on Tracking

```bash
# Run tracking tests
python -m pytest tests/track/ -v

# Debug pipeline with VSCode
# Use "🎯 Track: Debug Pipeline" configuration
```

#### 🧠 Working on Training

```bash
# Quick training run
python src/scripts/train_workflow.py --tenant_id tenant1 --num_epochs 3

# Debug training with VSCode
# Use "🧠 Train: Debug Training Script" configuration
```

#### 🌐 Working on API

```bash
# Start development server
python src/api/v1/main.py

# Debug API with VSCode
# Use "🌐 API: Debug Server" configuration
```

## 🧪 Testing Strategy

### Service-Specific Tests

- **Track Tests**: `tests/track/` - Pipeline and tracking functionality
- **Train Tests**: `tests/train/` - Training and model tests
- **API Tests**: `tests/api/` - API endpoint tests

### Running Tests

- **All Tests**: Use `🧪 Test: Run All Tests` configuration
- **Service Tests**: Use service-specific test configurations
- **Current File**: Use `🧪 Test: Debug Current File` for the active test file

## 🐳 Docker Integration

### Available Tasks

- **Build Worker**: `🐳 Docker: Build Worker Image`
- **Build API**: `🐳 Docker: Build API Image`

### Service Containers

- **Worker** (`docker/worker/`) - Processing and training workloads
- **API** (`docker/thin_api/`) - Lightweight API server

## ⚙️ Configuration

### Environment Variables

All configurations require:

```bash
PYTHONPATH="${workspaceFolder}/src"
GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
GOOGLE_CLOUD_PROJECT="laxai-466119"
```

### Python Path

The workspace is configured to include all service paths for proper imports:

- `./src`
- `./src/track`
- `./src/train`
- `./src/api`
- `./src/cloud`
- `./src/common`
- `./src/config`

## 🔧 Development Tips

1. **Service Isolation**: Each service can be developed and tested independently
2. **Shared Code**: Common functionality lives in `src/common/` and `src/config/`
3. **Debug Configs**: Use service-specific debug configurations for focused development
4. **Task Runner**: Use VS Code tasks for common operations (testing, building, etc.)
5. **Testing**: Run service-specific tests to avoid interference between services

## 📊 Workspace Features

- **🎨 Emojis**: Service sections are clearly marked with emojis for easy navigation
- **🎯 Focused Folders**: Each service appears as a separate workspace folder
- **🚀 Launch Configs**: Service-specific debug and run configurations
- **⚡ Tasks**: Pre-configured tasks for common operations
- **🧪 Testing**: Integrated test runner with service-specific test suites

## 🚀 Next Steps

1. Install recommended VS Code extensions for the best experience
2. Explore the debug configurations in the "Run and Debug" panel
3. Try the built-in tasks using `Ctrl+Shift+P` → "Tasks: Run Task"
4. Use the service-specific folder structure to navigate efficiently

Happy coding! 🎉
