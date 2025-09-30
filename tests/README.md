# LaxAI Test Organization

This directory contains the complete test suite for the LaxAI project, organized in a hierarchical structure for clarity and maintainability.

## Directory Structure

```
tests/
├── shared/                      # Tests for shared_libs components
│   ├── unit/                    # Unit tests for shared utilities
│   ├── integration/             # Integration tests for shared components
│   └── fixtures/                # Shared test fixtures
├── services/                    # Service-specific tests
│   ├── api/                     # API service tests
│   │   ├── unit/
│   │   ├── integration/
│   │   └── fixtures/
│   ├── tracking/                # Tracking service tests
│   │   ├── unit/
│   │   ├── integration/
│   │   └── fixtures/
│   ├── training/                # Training service tests
│   │   ├── unit/
│   │   ├── integration/
│   │   ├── performance/
│   │   └── fixtures/
│   └── dataprep/                # Data preparation service tests
│       ├── unit/
│       ├── integration/
│       └── fixtures/
└── system/                      # End-to-end system tests
    ├── integration/             # System integration tests
    ├── performance/             # System performance tests
    └── fixtures/                # System-level fixtures
```

## Test Categories

### 🔧 Shared Tests (`tests/shared/`)
Tests for components in `shared_libs/` that are used across multiple services.

- **Unit Tests**: Test individual shared utilities in isolation
- **Integration Tests**: Test shared components working together

### 🏗️ Service Tests (`tests/services/`)
Tests specific to individual microservices.

- **Unit Tests**: Test service components in isolation
- **Integration Tests**: Test service components working together
- **Performance Tests**: Service-specific performance and load tests

### 🌐 System Tests (`tests/system/`)
End-to-end tests that span multiple services.

- **Integration Tests**: Full system integration tests
- **Performance Tests**: System-wide performance benchmarks

## Running Tests

```bash
# Run all tests
pytest

# Run specific categories
pytest tests/shared/
pytest tests/services/
pytest tests/system/

# Run by service
pytest tests/services/training/
pytest tests/services/tracking/
pytest tests/services/api/

# Run by test type
pytest tests/shared/unit/
pytest tests/services/training/integration/
pytest tests/system/performance/

# Run with markers
pytest -m "integration and not slow"
pytest -m performance
```

## Test Markers

- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.skipif` - Conditional test skipping

## Best Practices

1. **Unit Tests**: Use mocks, avoid external dependencies
2. **Integration Tests**: Test real component interactions within service boundaries
3. **System Tests**: Test end-to-end workflows across services
4. **Performance Tests**: Focus on metrics, may require special setup
5. **Clear Boundaries**: Keep service tests focused on their service, use system tests for cross-service testing

## Adding New Tests

1. Determine the appropriate category (shared/service/system)
2. Choose the correct test type (unit/integration/performance)
3. Place in the corresponding directory
4. Use descriptive names: `test_feature_description.py`
5. Add appropriate pytest markers
6. Include docstrings explaining test purpose</content>
<parameter name="filePath">/Users/fernandomousinho/Library/Mobile Documents/com~apple~CloudDocs/Documents/Learning_to_Code/LaxAI/tests/README.md