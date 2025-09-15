#!/usr/bin/env python3
"""
Standalone script to clean up WandB test collections and artifacts.
Run this to clean up collections left behind by tests.
"""

import os
import sys

# Add shared_libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared_libs'))

import wandb
from shared_libs.config.all_config import wandb_config
from shared_libs.utils.env_secrets import setup_environment_secrets


def cleanup_wandb_test_artifacts(api, project_name, run_name=None):
    """
    Comprehensive cleanup of WandB test artifacts and collections.
    
    Args:
        api: WandB API instance
        project_name: Name of the WandB project
        run_name: Optional specific run name to clean up (defaults to cleaning all test- prefixed items)
    
    Returns:
        int: Total number of items deleted
    """
    total_deleted = 0
    collections_to_delete = []
    
    try:
        print(f"🔍 Starting WandB cleanup for project: {project_name}")
        
        # Get all artifact types
        artifact_types = api.artifact_types(project=project_name)
        print(f"📋 Found {len(list(artifact_types))} artifact types")

        # Re-fetch artifact types since the list was consumed
        artifact_types = api.artifact_types(project=project_name)

        for artifact_type in artifact_types:
            try:
                print(f"🔍 Checking artifact type: {artifact_type.name}")
                # Get all collections for this artifact type
                collections = list(artifact_type.collections())
                print(f"📁 Found {len(collections)} collections in {artifact_type.name}")

                for collection in collections:
                    try:
                        print(f"🔍 Examining collection: {collection.name}")
                        
                        # Conservative cleanup criteria - only target explicit test collections
                        should_cleanup = (
                            "test" in collection.name.lower() or
                            (run_name and run_name in collection.name)
                        )
                        
                        if should_cleanup:
                            print(f"🎯 Marked for cleanup: {collection.name}")
                            artifacts = list(collection.artifacts())
                            print(f"📦 Collection has {len(artifacts)} artifacts")
                            
                            artifacts_deleted_from_collection = 0

                            # Delete all artifacts in the collection
                            for artifact in artifacts:
                                try:
                                    # Safety check: Never delete artifacts with "do not delete" tag/alias
                                    if hasattr(artifact, 'aliases') and artifact.aliases:
                                        aliases = list(artifact.aliases) if artifact.aliases else []
                                        if any("do not delete" in str(alias).lower() for alias in aliases):
                                            print(f"🛡️ Skipping artifact {artifact.name} - has 'do not delete' protection")
                                            continue
                                    
                                    print(f"🗑️ Attempting to delete artifact: {artifact.name}")
                                    artifact.delete()
                                    print(f"✅ Deleted artifact: {artifact.name}")
                                    total_deleted += 1
                                    artifacts_deleted_from_collection += 1
                                except Exception as e:
                                    error_msg = str(e).lower()
                                    print(f"⚠️ Failed to delete artifact {artifact.name}: {e}")
                                    
                                    # Try more aggressive cleanup for aliased artifacts
                                    if "alias" in error_msg or "409" in str(e) or "conflict" in error_msg:
                                        try:
                                            print(f"🔄 Attempting alias removal for {artifact.name}")
                                            
                                            # Try to get and remove all aliases
                                            if hasattr(artifact, 'aliases'):
                                                try:
                                                    aliases = list(artifact.aliases) if artifact.aliases else []
                                                    print(f"🏷️ Found {len(aliases)} aliases: {aliases}")
                                                    
                                                    for alias in aliases:
                                                        try:
                                                            # Try different methods to remove alias
                                                            artifact.remove_alias(alias)
                                                            print(f"✅ Removed alias '{alias}' from {artifact.name}")
                                                        except Exception as alias_error:
                                                            print(f"⚠️ Failed to remove alias '{alias}': {alias_error}")
                                                            try:
                                                                # Alternative alias removal method
                                                                if alias in artifact.aliases:
                                                                    artifact.aliases.remove(alias)
                                                                    artifact.save()
                                                                print(f"✅ Removed alias '{alias}' via list method")
                                                            except Exception as alt_error:
                                                                print(f"⚠️ Alternative alias removal failed: {alt_error}")
                                                
                                                except Exception as alias_access_error:
                                                    print(f"⚠️ Failed to access aliases: {alias_access_error}")

                                            # Try to delete again after alias cleanup
                                            artifact.delete()
                                            print(f"✅ Deleted artifact after alias cleanup: {artifact.name}")
                                            total_deleted += 1
                                            artifacts_deleted_from_collection += 1
                                            
                                        except Exception as retry_error:
                                            print(f"❌ Final delete attempt failed for {artifact.name}: {retry_error}")
                                    else:
                                        print(f"❌ Skipping artifact {artifact.name} due to error: {e}")

                            # Mark collection for deletion regardless of artifacts remaining
                            # WandB should allow deleting collections even with artifacts in some cases
                            collections_to_delete.append((artifact_type, collection))
                        else:
                            print(f"⏭️ Skipping collection: {collection.name}")

                    except Exception as e:
                        print(f"❌ Failed to process collection {collection.name}: {e}")

            except Exception as e:
                print(f"❌ Failed to process artifact type {artifact_type.name}: {e}")

        # Now attempt to delete collections
        print(f"🗂️ Attempting to delete {len(collections_to_delete)} collections")
        for artifact_type, collection in collections_to_delete:
            try:
                print(f"🗑️ Attempting to delete collection: {collection.name}")
                
                # Check remaining artifacts
                try:
                    remaining_artifacts = list(collection.artifacts())
                    print(f"📦 Collection {collection.name} has {len(remaining_artifacts)} remaining artifacts")
                except Exception as artifact_check_error:
                    print(f"⚠️ Could not check remaining artifacts: {artifact_check_error}")
                    remaining_artifacts = []

                # Try to delete the collection
                collection.delete()
                print(f"✅ Deleted collection: {collection.name}")
                total_deleted += 1
                
            except Exception as e:
                error_msg = str(e).lower()
                if "not empty" in error_msg or "artifacts" in error_msg:
                    print(f"⚠️ Collection {collection.name} not empty, cannot delete: {e}")
                else:
                    print(f"⚠️ Failed to delete collection {collection.name}: {e}")

        print(f"✅ Cleanup completed. Total items deleted: {total_deleted}")

    except Exception as e:
        print(f"❌ Cleanup failed with error: {e}")

    return total_deleted


def list_all_collections(api, project_name):
    """List all collections in the project to see what exists."""
    print(f"📋 Listing all collections in project: {project_name}")
    
    try:
        artifact_types = api.artifact_types(project=project_name)
        
        for artifact_type in artifact_types:
            try:
                collections = list(artifact_type.collections())
                print(f"\n🏷️ Artifact Type: {artifact_type.name}")
                print(f"   Collections ({len(collections)}):")
                
                for collection in collections:
                    try:
                        artifacts = list(collection.artifacts())
                        print(f"   - {collection.name} ({len(artifacts)} artifacts)")
                    except Exception as e:
                        print(f"   - {collection.name} (error counting artifacts: {e})")
                        
            except Exception as e:
                print(f"❌ Failed to list collections for {artifact_type.name}: {e}")
                
    except Exception as e:
        print(f"❌ Failed to list collections: {e}")


def main():
    """Main function to run cleanup."""
    print("🧹 WandB Test Artifact Cleanup Tool")
    print("=" * 50)
    
    # Setup environment
    setup_environment_secrets()
    
    # Initialize WandB API
    api = wandb.Api()
    project_name = wandb_config.project
    
    print(f"🎯 Target project: {project_name}")
    
    # First, list all collections
    print("\n1️⃣ Listing all existing collections...")
    list_all_collections(api, project_name)
    
    # Ask for confirmation
    print("\n❓ Do you want to proceed with cleanup? (y/N): ", end="")
    response = input().strip().lower()
    
    if response == 'y' or response == 'yes':
        print("\n2️⃣ Starting cleanup...")
        deleted_count = cleanup_wandb_test_artifacts(api, project_name)
        
        print(f"\n✅ Cleanup completed! Deleted {deleted_count} items")
        
        print("\n3️⃣ Listing collections after cleanup...")
        list_all_collections(api, project_name)
    else:
        print("❌ Cleanup cancelled by user")


if __name__ == "__main__":
    main()