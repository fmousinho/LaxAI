# LaxAI Multi-Service Architecture

## Structure

```
LaxAI/
├── .vscode/                    # VS Code workspace settings
├── services/                   # Individual microservices
│   ├── service-tracking/       # Video processing and tracking
│   ├── service-training/       # ML training and model ops
│   ├── service-api/           # REST API endpoints
│   └── service-cloud/         # Cloud functions and batch processing
├── shared_libs/               # Code shared between services
│   ├── common/
│   ├── config/
│   └── utils/
├── tests/                     # Integration tests across services
└── docs/                      # Project documentation
```

## Services

### Service-Tracking

Handles video processing and player tracking operations.

### Service-Training

Handles model training, evaluation, and ML pipeline operations.

### Service-API

Handles REST API endpoints and web interface operations.

### Service-Cloud

Handles cloud functions, batch processing, and distributed operations.

## Development

Each service is self-contained with its own:

- Source code (`src/`)
- Tests (`tests/`)
- Dependencies (`requirements.txt`)
- Deployment configuration (`Dockerfile`)
- Virtual environment (`.venv/`)

## Getting Started

1. Navigate to the service you want to work on:

   ```bash
   cd services/service-tracking
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the service:
   ```bash
   python src/main.py
   ```
