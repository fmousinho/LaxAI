#!/usr/bin/env python3
"""
Test configuration switcher for LaxAI
Helps switch between fast test configuration and production configuration
"""

import os
import shutil
import sys
from pathlib import Path

def switch_to_test_config():
    """Switch to test configuration for faster local testing"""
    config_path = Path("config.toml")
    test_config_path = Path("test_config.toml")
    backup_path = Path("config.toml.backup")

    if not test_config_path.exists():
        print("❌ test_config.toml not found!")
        return False

    if not config_path.exists():
        print("❌ config.toml not found!")
        return False

    # Backup current config
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
        print("📁 Backed up config.toml to config.toml.backup")

    # Switch to test config
    shutil.copy2(test_config_path, config_path)
    print("✅ Switched to test configuration")
    print("   - Smaller batch sizes")
    print("   - Reduced workers")
    print("   - Optimized for local testing")
    return True

def switch_to_prod_config():
    """Switch back to production configuration"""
    config_path = Path("config.toml")
    backup_path = Path("config.toml.backup")

    if not backup_path.exists():
        print("❌ No backup found! Run with --test first to create backup")
        return False

    # Restore production config
    shutil.copy2(backup_path, config_path)
    print("✅ Switched back to production configuration")
    return True

def show_current_config():
    """Show current configuration status"""
    config_path = Path("config.toml")
    test_config_path = Path("test_config.toml")
    backup_path = Path("config.toml.backup")

    print("📊 Configuration Status:")
    print(f"   config.toml: {'✅' if config_path.exists() else '❌'}")
    print(f"   test_config.toml: {'✅' if test_config_path.exists() else '❌'}")
    print(f"   config.toml.backup: {'✅' if backup_path.exists() else '❌'}")

    if config_path.exists():
        with open(config_path, 'r') as f:
            content = f.read()
            if 'batch_size = 8' in content:
                print("   Current mode: 🧪 TEST (fast)")
            elif 'batch_size = 256' in content:
                print("   Current mode: 🚀 PRODUCTION")
            else:
                print("   Current mode: ❓ UNKNOWN")

def main():
    if len(sys.argv) != 2:
        print("Usage: python switch_config.py [--test|--prod|--status]")
        print("")
        print("Options:")
        print("  --test    Switch to test configuration (faster for local development)")
        print("  --prod    Switch back to production configuration")
        print("  --status  Show current configuration status")
        return

    command = sys.argv[1]

    if command == "--test":
        switch_to_test_config()
    elif command == "--prod":
        switch_to_prod_config()
    elif command == "--status":
        show_current_config()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use --test, --prod, or --status")

if __name__ == "__main__":
    main()
