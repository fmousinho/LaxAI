This folder contains the project's tests, including end-to-end (e2e) training tests.

Quick instructions to run the e2e tests locally:

1. Create and activate the project's virtualenv (the project uses `.venv31211` in the repo):

   python -m venv .venv31211
   source .venv31211/bin/activate

2. Install dev dependencies:

   pip install -U pip
   pip install -e .[dev]

3. Run the specific e2e test (this will use real GCP and wandb as the test expects):

   python -m pytest tests/test_train_all_e2e.py::test_train_all_with_one_dataset -q

To run tests from VS Code Test Explorer:

- Ensure the Python extension is installed.
- Open the Testing panel (Activity Bar). The extension will use the interpreter set in `.vscode/settings.json`.
- Click the refresh icon if tests are not discovered automatically.

Notes:
- The e2e test interacts with real external services (GCP, wandb). Make sure required credentials are available in your environment before running.
- Common env vars: `WANDB_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS` (or run `gcloud auth application-default login`).
