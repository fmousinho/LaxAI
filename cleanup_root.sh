#!/bin/bash
# Root Directory Cleanup Script
# Run this after verifying the multi-service migration is complete

echo "🧹 LaxAI Root Directory Cleanup"
echo "==============================="

# Create backup directory
mkdir -p ./backup_removed_files

# Function to safely remove and backup
safe_remove() {
    local file_or_dir="$1"
    if [ -e "$file_or_dir" ]; then
        echo "📦 Backing up and removing: $file_or_dir"
        mv "$file_or_dir" ./backup_removed_files/
    else
        echo "⚠️  Not found: $file_or_dir"
    fi
}

echo "🗂️  Removing old application files..."
safe_remove "application.py"
safe_remove "main.py old"
safe_remove "pyproject.toml.old"

echo "🗂️  Removing migrated directories..."
safe_remove "src"
safe_remove "modules"

echo "🗂️  Removing old virtual environments..."
safe_remove ".venv31211"

echo "🗂️  Removing build artifacts..."
safe_remove "__pycache__"
safe_remove ".pytest_cache"
safe_remove ".coverage"
safe_remove "dist"
safe_remove "build"

echo "🗂️  Removing old logs and results..."
safe_remove "wandb"
safe_remove "evaluation_results"

echo "🗂️  Removing old workspace files..."
safe_remove "lacrosse-ai.code-workspace"
safe_remove "LaxAI-multiservice.code-workspace"

echo "🗂️  Removing old config..."
safe_remove "cloud-config.env"

echo "🗂️  Removing old deployment scripts..."
safe_remove "scripts"

echo "🗂️  Removing old requirements..."
safe_remove "requirements"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📋 Files to manually review and potentially move/update:"
echo "   - run_tests.py (update for service architecture)"
echo "   - switch_config.py (update for service configs)"
echo "   - tools/ (evaluate and move to appropriate services)"
echo "   - tests/ (reorganize: integration stays, unit tests to services)"
echo ""
echo "💾 All removed files have been backed up to './backup_removed_files/'"
echo "🗑️  You can delete the backup directory after verifying everything works."
