#!/usr/bin/env python3
"""
Multi-Service Configuration Switcher for LaxAI
Helps switch between test/dev and production configurations across all services
"""

import os
import shutil
import sys
from pathlib import Path


def switch_service_config(service_name, to_test=True):
    """Switch configuration for a specific service"""
    service_path = Path(f"services/{service_name}")
    if not service_path.exists():
        print(f"⚠️  Service {service_name} not found, skipping...")
        return False

    config_path = service_path / "config.toml"
    test_config_path = service_path / "test_config.toml"
    backup_path = service_path / "config.toml.backup"

    if to_test:
        if not test_config_path.exists():
            print(f"❌ {service_name}: test_config.toml not found!")
            return False

        if not config_path.exists():
            print(f"❌ {service_name}: config.toml not found!")
            return False

        # Backup current config
        if not backup_path.exists():
            shutil.copy2(config_path, backup_path)
            print(f"📁 {service_name}: Backed up config.toml")

        # Switch to test config
        shutil.copy2(test_config_path, config_path)
        print(f"🔄 {service_name}: Switched to test configuration")
    else:
        if backup_path.exists():
            shutil.copy2(backup_path, config_path)
            print(f"🔄 {service_name}: Restored original configuration")
        else:
            print(f"⚠️  {service_name}: No backup found, cannot restore")
            return False

    return True

    # Switch to test config
    shutil.copy2(test_config_path, config_path)
    print("✅ Switched to test configuration")
    print("   - Smaller batch sizes")
    print("   - Reduced workers")
    print("   - Optimized for local testing")
    return True

def switch_to_prod_config():
    """Switch back to production configuration"""
def switch_to_test_config():
    """Switch all services to test configuration for faster local testing"""
    services = ['service_tracking', 'service_training', 'service_cloud']
    success_count = 0
    
    # Also handle root config if it exists
    config_path = Path("config.toml")
    test_config_path = Path("test_config.toml")
    backup_path = Path("config.toml.backup")

    if test_config_path.exists() and config_path.exists():
        if not backup_path.exists():
            shutil.copy2(config_path, backup_path)
            print("📁 Root: Backed up config.toml")
        shutil.copy2(test_config_path, config_path)
        print("🔄 Root: Switched to test configuration")
        success_count += 1

    # Switch each service
    for service in services:
        if switch_service_config(service, to_test=True):
            success_count += 1

    print(f"\n✅ Successfully switched {success_count} configurations to test mode")
    return success_count > 0

def switch_to_prod_config():
    """Switch all services back to production configuration"""
    services = ['service_tracking', 'service_training', 'service_cloud']
    success_count = 0
    
    # Handle root config
    config_path = Path("config.toml")
    backup_path = Path("config.toml.backup")

    if backup_path.exists():
        shutil.copy2(backup_path, config_path)
        print("🔄 Root: Restored original configuration")
        success_count += 1

    # Switch each service back
    for service in services:
        if switch_service_config(service, to_test=False):
            success_count += 1

    print(f"\n✅ Successfully restored {success_count} configurations to production")
    return success_count > 0

def show_current_config():
    """Show current configuration status for all services"""
    services = ['service_tracking', 'service_training', 'service_cloud']
    
    print("📊 Multi-Service Configuration Status:")
    print("=" * 50)
    
    # Check root config
    config_path = Path("config.toml")
    test_config_path = Path("test_config.toml")
    backup_path = Path("config.toml.backup")

    print("🏠 Root Configuration:")
    print(f"   config.toml: {'✅' if config_path.exists() else '❌'}")
    print(f"   test_config.toml: {'✅' if test_config_path.exists() else '❌'}")
    print(f"   config.toml.backup: {'✅' if backup_path.exists() else '❌'}")

    # Check each service
    for service in services:
        service_path = Path(f"services/{service}")
        service_config = service_path / "config.toml"
        service_test_config = service_path / "test_config.toml"  
        service_backup = service_path / "config.toml.backup"
        
        print(f"\n📦 {service}:")
        print(f"   config.toml: {'✅' if service_config.exists() else '❌'}")
        print(f"   test_config.toml: {'✅' if service_test_config.exists() else '❌'}")
        print(f"   config.toml.backup: {'✅' if service_backup.exists() else '❌'}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python switch_config.py [--test|--prod|--status]")
        print("")
        print("Options:")
        print("  --test    Switch all services to test configuration (faster for local development)")
        print("  --prod    Switch all services back to production configuration")
        print("  --status  Show current configuration status for all services")
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
        return

if __name__ == "__main__":
    main()
