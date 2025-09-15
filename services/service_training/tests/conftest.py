"""
Shared test utilities and fixtures for the LaxAI project.
"""
import os
import sys
import time
from pathlib import Path

import pytest


def setup_test_paths():
    """
    Set up Python paths for test files that need to import from shared_libs and other modules.

    This function ensures that:
    - The project root is in sys.path
    - The shared_libs directory is in sys.path
    - The services directory is in sys.path

    This is necessary when running tests from VSCode testing browser or other environments
    where the PYTHONPATH may not be properly configured.
    """
    # Find the project root (assuming this file is in tests/conftest.py)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Go up two levels from tests/

    # Define important paths
    shared_libs_path = project_root / "shared_libs"
    services_path = project_root / "services"

    # Add paths to sys.path if not already present
    paths_to_add = [project_root, shared_libs_path, services_path]

    for path in paths_to_add:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


# Automatically set up paths when this module is imported
setup_test_paths()


def cleanup_wandb_test_artifacts(api, project_name, run_name=None):
    """
    Comprehensive cleanup of WandB test artifacts and collections.
    Deletes runs first, then artifacts, then collections in the correct order.
    Also cleans up model registry collections.
    
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
        
        # Step 1: Clean up model registry collections first
        print("🏛️ Cleaning up test model registry collections...")
        try:
            # Get the entity name from the API
            entity = api.default_entity
            print(f"🔍 Checking model registry for entity: {entity}")
            
            # Access the model registry project
            registry_project = f"{entity}-org/wandb-registry-model"
            print(f"🔍 Accessing registry project: {registry_project}")
            
            # Get artifact types from the registry
            registry_artifact_types = api.artifact_types(project=registry_project)
            for artifact_type in registry_artifact_types:
                print(f"🔍 Checking registry artifact type: {artifact_type.name}")
                collections = list(artifact_type.collections())
                
                for collection in collections:
                    try:
                        if collection.name.lower().startswith("test"):
                            print(f"🎯 Found test collection in registry: {collection.name}")
                            
                            # Delete artifacts in the collection first
                            artifacts = list(collection.artifacts())
                            print(f"📦 Collection has {len(artifacts)} artifacts")
                            
                            for artifact in artifacts:
                                try:
                                    # Safety check: Never delete artifacts with "do not delete" alias
                                    if hasattr(artifact, 'aliases') and artifact.aliases:
                                        aliases = list(artifact.aliases) if artifact.aliases else []
                                        if any("do not delete" in str(alias).lower() for alias in aliases):
                                            print(f"🛡️ Skipping artifact {artifact.name} - has 'do not delete' protection")
                                            continue
                                    
                                    print(f"🗑️ Deleting registry artifact: {artifact.name}")
                                    artifact.delete()
                                    print(f"✅ Deleted registry artifact: {artifact.name}")
                                    total_deleted += 1
                                except Exception as e:
                                    print(f"⚠️ Failed to delete registry artifact {artifact.name}: {e}")
                            
                            # Delete the collection
                            try:
                                print(f"🗑️ Deleting registry collection: {collection.name}")
                                collection.delete()
                                print(f"✅ Deleted registry collection: {collection.name}")
                                total_deleted += 1
                            except Exception as e:
                                print(f"⚠️ Failed to delete registry collection {collection.name}: {e}")
                        else:
                            print(f"⏭️ Skipping registry collection: {collection.name}")
                            
                    except Exception as e:
                        print(f"⚠️ Failed to process registry collection {collection.name}: {e}")
                    
        except Exception as e:
            print(f"⚠️ Failed to access model registry: {e}")
        
        # Step 2: Delete runs (runs must be deleted before their artifacts can be deleted)
        print("🏃 Cleaning up test runs...")
        try:
            runs = api.runs(project_name)
            for run in runs:
                try:
                    # Only delete runs that start with "test" (case insensitive)
                    if run.name and run.name.lower().startswith("test"):
                        # Check for "do not delete" tags
                        if run.tags and any("do not delete" in str(tag).lower() for tag in run.tags):
                            print(f"🛡️ Skipping run {run.name} - has 'do not delete' tag")
                            continue
                        
                        print(f"🗑️ Deleting run: {run.name}")
                        run.delete()
                        print(f"✅ Deleted run: {run.name}")
                        total_deleted += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete run {run.name}: {e}")
        except Exception as e:
            print(f"⚠️ Failed to access runs: {e}")
        
        # Step 3: Delete artifacts and collections from project
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
                            collection.name.lower().startswith("test") or
                            (run_name and run_name in collection.name)
                        )
                        
                        if should_cleanup:
                            print(f"🎯 Marked for cleanup: {collection.name}")
                            artifacts = list(collection.artifacts())
                            print(f"📦 Collection has {len(artifacts)} artifacts")
                            
                            # Delete all artifacts in the collection
                            for artifact in artifacts:
                                try:
                                    # Safety check: Never delete artifacts with "do not delete" alias
                                    if hasattr(artifact, 'aliases') and artifact.aliases:
                                        aliases = list(artifact.aliases) if artifact.aliases else []
                                        if any("do not delete" in str(alias).lower() for alias in aliases):
                                            print(f"🛡️ Skipping artifact {artifact.name} - has 'do not delete' protection")
                                            continue
                                    
                                    print(f"🗑️ Attempting to delete artifact: {artifact.name}")
                                    artifact.delete()
                                    print(f"✅ Deleted artifact: {artifact.name}")
                                    total_deleted += 1
                                except Exception as e:
                                    print(f"⚠️ Failed to delete artifact {artifact.name}: {e}")

                            # Mark collection for deletion
                            collections_to_delete.append((artifact_type, collection))
                        else:
                            print(f"⏭️ Skipping collection: {collection.name}")

                    except Exception as e:
                        print(f"❌ Failed to process collection {collection.name}: {e}")

            except Exception as e:
                print(f"❌ Failed to process artifact type {artifact_type.name}: {e}")

        # Step 3: Delete collections (after artifacts are deleted)
        print(f"🗂️ Attempting to delete {len(collections_to_delete)} collections")
        for artifact_type, collection in collections_to_delete:
            try:
                print(f"🗑️ Attempting to delete collection: {collection.name}")
                collection.delete()
                print(f"✅ Deleted collection: {collection.name}")
                total_deleted += 1
            except Exception as e:
                print(f"⚠️ Failed to delete collection {collection.name}: {e}")

        print(f"✅ Cleanup completed. Total items deleted: {total_deleted}")

    except Exception as e:
        print(f"❌ Cleanup failed with error: {e}")

    return total_deleted


@pytest.fixture(autouse=True)
def cleanup_wandb_after_test(request):
    """
    Pytest fixture that automatically cleans up WandB test artifacts after each test.
    
    This fixture runs after every test and cleans up:
    - Runs whose names start with "test"
    - Artifacts in collections whose names start with "test"  
    - Collections whose names start with "test"
    
    It respects "do not delete" tags/aliases and skips the test that needs persistence.
    """
    # Yield to let the test run first
    yield
    
    # Skip cleanup for tests that need artifacts to persist for resume functionality
    test_name = request.node.name
    if "resume" in test_name.lower() and "device" in test_name.lower():
        print(f"⏭️ Skipping cleanup for {test_name} - resume test needs artifact persistence")
        return
    
    try:
        # Import wandb and config after setup_test_paths has run
        import wandb
        from shared_libs.config.all_config import wandb_config
        
        print(f"🧹 Starting post-test cleanup for: {test_name}")
        
        # Wait for wandb synchronization before cleanup
        print("⏳ Waiting 5 seconds for wandb synchronization...")
        time.sleep(5)
        
        try:
            api = wandb.Api()
            cleanup_wandb_test_artifacts(api, wandb_config.project)
            print(f"✅ Post-test cleanup completed for: {test_name}")
        except Exception as e:
            print(f"⚠️ Failed to initialize wandb API for cleanup: {e}")
            
    except ImportError as e:
        print(f"⚠️ Could not import wandb for cleanup: {e}")
    except Exception as e:
        print(f"⚠️ Post-test cleanup failed for {test_name}: {e}")
