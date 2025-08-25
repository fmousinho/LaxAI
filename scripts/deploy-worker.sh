#!/bin/bash

# Wrapper that delegates to the longer deploy-worker.sh implementation in the repo root
# (keeps backward compatibility with existing top-level script names)

set -e

if [[ -f ./deploy-worker.sh ]]; then
  exec ./deploy-worker.sh "$@"
else
  echo "deploy-worker.sh not found at repo root; fallback to scripts/deploy-worker.sh logic is recommended."
  exit 1
fi
