# Shared Libraries

Common utilities and libraries used across multiple services.

## Structure

- `common/` - Common utilities and base classes
- `config/` - Configuration management
- `utils/` - Utility functions shared across services

## Usage

These libraries can be imported by services that need shared functionality.
Each service should still maintain its own requirements.txt for dependencies.
