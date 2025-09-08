

# utils/__init__.py
# Expose utility functions for environment setup, wandb uploading, and ID generation

try:
    from .env_secrets import setup_environment_secrets
except Exception:
    # Defer import errors until the function is used
    setup_environment_secrets = None

try:
    from .id_generator import (
    create_tenant_id,
    create_video_id,
    create_frame_id,
    create_dataset_id,
    create_run_id,
    create_user_id,
    create_experiment_id,
    create_model_id,
    create_crop_id,
    create_batch_id,
    create_full_guid,
    create_short_guid
)
except Exception:
    # id_generator may be unavailable during isolated tests; ignore
    pass

#from .wandb_uploader import wandb_uploader
