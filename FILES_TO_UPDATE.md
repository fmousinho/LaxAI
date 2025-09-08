# Files that need manual updates/migration

## run_tests.py

- **Current**: Monolithic test runner for all tests
- **Action Needed**: Update to work with service-specific test suites
- **Suggestion**: Create a root-level test runner that:
  1. Runs tests for each service independently
  2. Runs integration tests between services
  3. Provides unified reporting

## switch_config.py

- **Current**: Switches between test and production config
- **Action Needed**: Update for service-specific configurations
- **Suggestion**: Either:
  1. Update to handle service-specific config switching, or
  2. Move to shared_libs as a utility for all services

## tools/

- **Contents**:
  - `reporting.py` - Reporting utilities
  - `test_checkpoint_fail.py` - Test utilities
  - `utils.py` - General utilities
  - `credentials.json` & `token.json` - Auth files
- **Action Needed**: Evaluate each file:
  - Move general utilities to `shared_libs/utils/`
  - Move service-specific tools to appropriate services
  - Keep auth files secure or move to proper config location

## tests/

- **Contents**:
  - `unit/` - Unit tests (move to individual services)
  - `integration/` - Keep at root for cross-service testing
  - `api/` - Move to `services/service-api/tests/`
  - `track/` - Move to `services/service-tracking/tests/`
  - `performance/` - Keep at root or move to shared testing
- **Action Needed**: Reorganize test structure to match services

## Config Files to Update

### pytest.ini

- Update test discovery paths for new service structure
- Add service-specific test markers

### pyrightconfig.json

- Update include/exclude paths for new structure
- Add service-specific type checking rules

### .vscode/ settings

- Update Python path settings to avoid PYTHONPATH conflicts
- Add service-specific debug configurations

## Deployment Files

### docker/

- Update Dockerfiles for individual services
- Create docker-compose for multi-service setup

### deploy/

- Update deployment scripts for service-specific deployments
- Add orchestration for multi-service deployment

### .github/workflows

- Update CI/CD for service-specific testing and deployment
- Add integration testing workflows

## Environment Files

### config.toml

- Keep for global/shared configuration
- Consider splitting into service-specific configs where appropriate

### .env files

- Update for service-specific environment variables
- Ensure proper secret management across services
