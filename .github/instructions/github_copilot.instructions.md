---
applyTo: "**"
---

When running a test, make sure you are using the correct .venv environment.

# LaxAI Services Overview

We are transitioning to a multi-service architecture for better modularity and scalability. Each service has its own codebase, dependencies, and tests.

All services share common libraries located in `shared_libs/`.

## Service Structure

Each service follows this structure:

```
service_name/
├── src/                    # Source code
├── tests/                  # Tests (unit, integration, performance)
├── requirements.txt        # Service-specific dependencies
├── Dockerfile              # Docker configuration
└── .venv/                  # Virtual environment for the service
```

## Shared Libraries

Common utilities and libraries used across multiple services are located in `shared_libs/`.
These include:

- `common/` - Common utilities and base classes
- `config/` - Configuration management
- `utils/` - Utility functions shared across services

## Running Tests

Each service has its own tests located in the `tests/` directory within the service folder.
To run tests for a specific service, navigate to the service directory and activate its virtual environment:

```bash
cd services/service_name
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pytest tests/ -v
```

Replace `service_name` with the actual service you want to test (e.g., `service_tracking`, `service_training`, etc.).

## General rules

- Imports should always use absolute paths from the service root or shared_libs.
- Each service should maintain its own `requirements.txt` for dependencies.
- Service tests should not depend on other services.
- Integration tests should only test interactions within the service and with shared_libs, not with other services.
- Performance tests should be marked with `@pytest.mark.performance` and can be run separately.
- sv-tests shoud contain all tests that verify the system as a whole, including inter-service interactions.
- There should not be an src folder in the root directory anymore.

## Code Review and Best Practices

- Do not accept requests blindly. If they don't follow best practices or if you see better alternatives, provide that critique before continuing.
- Always follow architecture best practices. This includes avoiding redundant/repeated code and having constants spread throughout the code.
- If you see opportunities to structure the code in a more elegant and easy to maintain way, you should suggest them before pursuing it.
- You don't need to generate documentation with every change. As a matter of fact, I will let you know if documentation is needed.
