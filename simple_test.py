#!/usr/bin/env python3
import os
import sys
sys.path.append('src')

print("Testing WandB Secret Manager integration...")

# Check before
print(f"Before: WANDB_API_KEY = {os.environ.get('WANDB_API_KEY')}")

# Import and see if it gets loaded
from utils.env_or_colab import load_env_or_colab

# Check after
print(f"After: WANDB_API_KEY = {os.environ.get('WANDB_API_KEY')}")

# Check if Secret Manager is available
try:
    from google.cloud import secretmanager
    print("Secret Manager library available: ✅")
except ImportError:
    print("Secret Manager library available: ❌")

# Check Google credentials
creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
if creds and os.path.exists(creds):
    print(f"Google credentials file: ✅ {creds}")
else:
    print(f"Google credentials file: ❌ {creds}")

print("Done.")
