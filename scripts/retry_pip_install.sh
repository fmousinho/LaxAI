#!/bin/bash
# Robust pip install script with retry logic for Docker builds
# Usage: ./scripts/retry_pip_install.sh requirements.txt

set -e

REQUIREMENTS_FILE="$1"
MAX_RETRIES=3
TIMEOUT=300

if [ -z "$REQUIREMENTS_FILE" ]; then
    echo "Usage: $0 <requirements-file>"
    exit 1
fi

echo "Installing requirements from $REQUIREMENTS_FILE with retry logic..."

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "Attempt $attempt of $MAX_RETRIES..."

    if pip install --no-cache-dir --timeout=$TIMEOUT --retries=10 -r "$REQUIREMENTS_FILE"; then
        echo "✅ Successfully installed requirements on attempt $attempt"
        exit 0
    else
        echo "❌ Attempt $attempt failed"

        if [ $attempt -lt $MAX_RETRIES ]; then
            sleep_time=$((5 * attempt))
            echo "⏳ Waiting ${sleep_time}s before retry..."
            sleep $sleep_time
        fi
    fi
done

echo "💥 All $MAX_RETRIES attempts failed"
exit 1