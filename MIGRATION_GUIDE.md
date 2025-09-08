# Migration Guide: Transitioning to Multi-Service Architecture

## Overview

This guide outlines how to migrate your existing LaxAI codebase to the new multi-service architecture.

## Current Structure Analysis

Your current structure has these main components:

- `src/track/` → Should move to `services/service-tracking/src/`
- `src/train/` → Should move to `services/service-training/src/`
- `src/api/` → Should move to `services/service-api/src/`
- `src/cloud/` → Should move to `services/service-cloud/src/`
- `src/common/`, `src/config/`, `src/utils/` → Should move to `shared_libs/`

## Migration Steps

### Phase 1: Service Separation

1. **Tracking Service Migration**:

   ```bash
   # Copy tracking-related code
   cp -r src/track/* services/service-tracking/src/
   cp -r modules/* services/service-tracking/src/  # Move modules to tracking service

   # Copy relevant tests
   cp -r tests/track/* services/service-tracking/tests/
   ```

2. **Training Service Migration**:

   ```bash
   # Copy training-related code
   cp -r src/train/* services/service-training/src/

   # Copy relevant tests
   cp -r tests/unit/test_*train* services/service-training/tests/
   cp -r tests/integration/test_training* services/service-training/tests/
   cp -r tests/performance/* services/service-training/tests/
   ```

3. **API Service Migration**:

   ```bash
   # Copy API-related code
   cp -r src/api/* services/service-api/src/
   cp -r src/services/* services/service-api/src/  # Move service layer to API

   # Copy relevant tests
   cp -r tests/api/* services/service-api/tests/
   ```

4. **Cloud Service Migration**:

   ```bash
   # Copy cloud-related code
   cp -r src/cloud/* services/service-cloud/src/

   # Copy relevant scripts
   cp -r scripts/* services/service-cloud/src/
   ```

### Phase 2: Shared Libraries

5. **Move Shared Code**:
   ```bash
   # Move common utilities
   cp -r src/common/* shared_libs/common/
   cp -r src/config/* shared_libs/config/
   cp -r src/utils/* shared_libs/utils/
   ```

### Phase 3: Environment Setup

6. **Create Service Virtual Environments**:
   ```bash
   # For each service
   cd services/service-tracking && python -m venv .venv
   cd services/service-training && python -m venv .venv
   cd services/service-api && python -m venv .venv
   cd services/service-cloud && python -m venv .venv
   ```

### Phase 4: Import Path Updates

7. **Update Import Statements**:
   - Change `from src.train.xyz` → `from xyz` (within training service)
   - Change `from src.common.xyz` → `from shared_libs.common.xyz`
   - Update relative imports to be service-specific

### Phase 5: Testing

8. **Update Test Configurations**:
   - Move `pytest.ini` to each service directory
   - Update test discovery paths
   - Ensure tests run independently per service

## Post-Migration Checklist

- [ ] Each service runs independently
- [ ] Each service has its own virtual environment
- [ ] All import paths are corrected
- [ ] Tests pass for each service
- [ ] Dockerfiles build successfully
- [ ] VS Code workspace opens correctly

## Benefits After Migration

1. **Independent Development**: Teams can work on different services without conflicts
2. **Isolated Dependencies**: No more dependency version conflicts between services
3. **Independent Deployment**: Services can be deployed and scaled separately
4. **Cleaner Testing**: Each service has focused, isolated tests
5. **Better Organization**: Clear separation of concerns

## Rollback Plan

If needed, you can rollback by:

1. Keeping the original `src/` structure intact during migration
2. Using git branches to track the migration process
3. Testing thoroughly before removing old structure
