"""
DataPrep Service SDK Manager

This module provides a Python SDK for the LaxAI dataprep service, exposing
functionality for track stitching verification workflows.
"""

import logging
from typing import List, Dict, Any, Optional

from shared_libs.common.google_storage import GCSPaths, get_storage
from shared_libs.common.detection_utils import load_all_detections_summary
from ..stitcher import TrackStitcher

logger = logging.getLogger(__name__)


class DataPrepManager:
    """
    Manager class for dataprep service operations.

    Provides methods to list process folders, start verification sessions,
    get images for verification, record user responses, and suspend/resume sessions.
    """

    def __init__(self, tenant_id: str):
        """
        Initialize the DataPrep manager for a specific tenant.

        Args:
            tenant_id: The tenant identifier (e.g., 'tenant1')
        """
        self.tenant_id = tenant_id
        self.path_manager = GCSPaths()
        self.storage = get_storage(tenant_id)
        self.stitcher: Optional[TrackStitcher] = None
        self.current_process_folder: Optional[str] = None

    def get_process_folders(self) -> List[str]:
        """
        Get a list of all process folder directories for the tenant.

        Returns:
            List of process folder names (video IDs)
        """
        # Get the process root path for this tenant
        process_root = self.path_manager.get_path("process_root")

        # List all blobs in the process root with delimiter to get directories
        try:
            folders = self.storage.list_blobs(
                prefix=process_root,
                delimiter='/',
                exclude_prefix_in_return=True
            )
            # Convert to list and filter out any non-directory items
            return list(folders)
        except Exception as e:
            logger.error(f"Failed to list process folders for tenant {self.tenant_id}: {e}")
            return []

    def start_prep(self, process_folder: str) -> bool:
        """
        Start a track stitching verification session for a process folder.

        Args:
            process_folder: The process folder name (video ID)

        Returns:
            True if successfully started, False otherwise
        """
        try:
            # Clean up the process folder name by removing .mp4 extension if present
            if process_folder.endswith('.mp4/'):
                process_folder = process_folder[:-5] + '/'  # Remove '.mp4/'
            elif process_folder.endswith('.mp4'):
                process_folder = process_folder[:-4]  # Remove '.mp4'
            
            # Get the detections.json path
            detections_path = self.path_manager.get_path(
                "detections_path",
                video_id=process_folder
            )
            if type(detections_path) is not str:
                raise ValueError(f"Invalid detections path: {detections_path}")

            # Load detections from GCS
            logger.info(f"Attempting to download detections from: {detections_path}")
            detections_json_text = self.storage.download_as_string(detections_path)
            if detections_json_text is None:
                raise ValueError(f"Could not download detections from {detections_path}. "
                               f"Please ensure the video has been processed by the tracking service first. "
                               f"Check that the process_folder name exactly matches the folder in GCS.")
            
            import json
            detections_json = json.loads(detections_json_text)
            
            # Load detections - the loader automatically handles different formats
            detections = load_all_detections_summary(detections_json)
            logger.info("Successfully loaded detections")

            # Check for existing saved graph
            existing_graph = None
            saved_graph_path = self.path_manager.get_path(
                "saved_graph",
                video_id=process_folder
            )
            
            if saved_graph_path is not None:
                try:
                    # Try to download the saved graph
                    graph_data = self.storage.download_as_string(saved_graph_path)
                    if graph_data is not None:
                        import networkx as nx
                        import io
                        # Load the graph from the downloaded data
                        existing_graph = nx.read_graphml(io.StringIO(graph_data))
                        logger.info(f"Found existing saved graph, will resume from {saved_graph_path}")
                    else:
                        logger.info("No existing saved graph found, starting fresh")
                except Exception as e:
                    logger.warning(f"Could not load existing saved graph: {e}, starting fresh")

            # Create the stitcher
            self.stitcher = TrackStitcher(detections=detections, existing_graph=existing_graph)
            self.current_process_folder = process_folder

            logger.info(f"Started prep session for process folder: {process_folder}")
            return True

        except Exception as e:
            logger.error(f"Failed to start prep for {process_folder}: {e}")
            return False

    def get_images_for_verification(self) -> Dict[str, Any]:
        """
        Get the next pair of track groups for verification.

        Returns:
            Dictionary containing:
            - status: 'pending_verification', 'complete', or 'second_pass_ready'
            - group1_id: First group ID (if status is 'pending_verification')
            - group2_id: Second group ID (if status is 'pending_verification')
            - group1_prefixes: List of GCS prefixes for group1 tracks
            - group2_prefixes: List of GCS prefixes for group2 tracks
        """
        if self.stitcher is None:
            return {"status": "error", "message": "No active stitcher session"}

        if self.current_process_folder is None:
            return {"status": "error", "message": "No current process folder"}

        try:
            result = self.stitcher.get_pair_for_verification()

            if result["status"] == "pending_verification":
                group1_id = result["group1_id"]
                group2_id = result["group2_id"]

                # Get track prefixes for each group
                group1_prefixes = self._get_group_track_prefixes(group1_id)
                group2_prefixes = self._get_group_track_prefixes(group2_id)

                progress_info = self.stitcher.get_verification_progress()

                return {
                    "status": "pending_verification",
                    "group1_id": group1_id,
                    "group2_id": group2_id,
                    "group1_prefixes": group1_prefixes,
                    "group2_prefixes": group2_prefixes,
                    "total_pairs": progress_info["total_pairs"],
                    "verified_pairs": progress_info["verified_pairs"],
                }
            else:
                if not self.save_graph():
                    logger.warning("Failed to save graph after completing verification")
                return result

        except Exception as e:
            logger.error(f"Failed to get images for verification: {e}")
            return {"status": "error", "message": str(e)}

    def record_response(self, decision: str) -> bool:
        """
        Record a user response for the current verification pair.

        Args:
            decision: User's decision - "same", "different", or "skip"

        Returns:
            True if response was recorded successfully, False otherwise
        """
        if self.stitcher is None:
            logger.error("No active stitcher session")
            return False

        try:
            self.stitcher.respond(decision)
            logger.info(f"Recorded response: {decision}")
            return True
        except Exception as e:
            logger.error(f"Failed to record response '{decision}': {e}")
            return False

    def _get_group_track_prefixes(self, group_id: int) -> List[str]:
        """
        Get GCS prefixes for all tracks in a group.

        Args:
            group_id: The group ID

        Returns:
            List of GCS prefixes for the tracks in the group
        """
        if self.stitcher is None or self.current_process_folder is None:
            return []

        # Get the tracks for this group
        group_tracks = self.stitcher.player_groups.get(group_id, [])

        prefixes = []
        for track_id in group_tracks:
            # Get the unverified_tracks path for this track
            prefix = self.path_manager.get_path(
                "unverified_tracks",
                video_id=self.current_process_folder,
                track_id=track_id
            )
            prefixes.append(prefix)

        return prefixes

    def save_graph(self) -> bool:
        """
        Save the current stitcher graph state to GCS.

        Returns:
            True if the graph was successfully saved, False otherwise
        """
        if self.stitcher is None:
            logger.error("No active stitcher session to save")
            return False

        if self.current_process_folder is None:
            logger.error("No current process folder set")
            return False

        import tempfile
        import os

        try:
            # Create a temporary file for the graph
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.graphml', delete=False) as temp_file:
                temp_filepath = temp_file.name

            # Save the graph to the temporary file
            success = self.stitcher.save_graph(temp_filepath, format="graphml")
            if not success:
                logger.error("Failed to save graph to temporary file")
                os.unlink(temp_filepath)
                return False

            # Get the GCS path for the saved graph
            gcs_path = self.path_manager.get_path(
                "saved_graph",
                video_id=self.current_process_folder
            )

            if gcs_path is None:
                logger.error("Failed to generate GCS path for saved graph")
                os.unlink(temp_filepath)
                return False

            # Upload the file to GCS
            upload_success = self.storage.upload_from_file(gcs_path, temp_filepath)

            # Clean up the temporary file
            os.unlink(temp_filepath)

            if upload_success:
                logger.info(f"Successfully saved graph to {gcs_path}")
                return True
            else:
                logger.error("Failed to upload graph to GCS")
                return False

        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            return False

    def save_graph_image(self) -> bool:
        """
        Generate and save a visual representation of the track relationship graph to GCS.

        Returns:
            True if the graph image was successfully saved, False otherwise
        """
        if self.stitcher is None:
            logger.error("No active stitcher session to visualize")
            return False

        if self.current_process_folder is None:
            logger.error("No current process folder set")
            return False

        import tempfile
        import os

        try:
            # Generate the visualization image
            image = self.stitcher.visualize_graph()
            if image is None:
                logger.error("Failed to generate graph visualization")
                return False

            # Create a temporary file for the image
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.jpg', delete=False) as temp_file:
                temp_filepath = temp_file.name

            # Save the PIL image to the temporary file
            image.save(temp_filepath, format='JPEG', quality=95)
            image.close()  # Free memory

            # Get the GCS path for the graph image
            gcs_path = self.path_manager.get_path(
                "track_graph_image",
                video_id=self.current_process_folder
            )

            if gcs_path is None:
                logger.error("Failed to generate GCS path for graph image")
                os.unlink(temp_filepath)
                return False

            # Upload the file to GCS
            upload_success = self.storage.upload_from_file(gcs_path, temp_filepath)

            # Clean up the temporary file
            os.unlink(temp_filepath)

            if upload_success:
                logger.info(f"Successfully saved graph visualization to {gcs_path}")
                return True
            else:
                logger.error("Failed to upload graph image to GCS")
                return False

        except Exception as e:
            logger.error(f"Error saving graph image: {e}")
            return False

    def suspend_prep(self) -> bool:
        """
        Save the current stitcher graph state to GCS for later resumption.

        Returns:
            True if the graph was successfully saved, False otherwise
        """
        return self.save_graph()

    def move_crops_to_verified(self) -> bool:
        """
        Move crops from unverified_tracks to verified_tracks based on track_graph associations.
        
        For each group in the track graph, consolidates all crops from tracks in that group
        into a single verified_tracks/{group_id}/ folder.

        Returns:
            True if all crops were moved successfully, False otherwise
        """
        if self.stitcher is None:
            logger.error("No active stitcher session")
            return False

        if self.current_process_folder is None:
            logger.error("No current process folder set")
            return False

        try:
            logger.info(f"Starting crop migration for video {self.current_process_folder}")
            moved_count = 0
            failed_count = 0

            # For each group in player_groups
            for group_id, track_ids in self.stitcher.player_groups.items():
                logger.info(f"Processing group {group_id} with tracks: {list(track_ids)}")
                
                # Collect all crop files from all tracks in this group
                crops_to_move = []
                
                for track_id in track_ids:
                    # Get the unverified_tracks path for this track
                    unverified_prefix = self.path_manager.get_path(
                        "unverified_tracks",
                        video_id=self.current_process_folder,
                        track_id=track_id
                    )
                    
                    if unverified_prefix is None:
                        logger.warning(f"Could not generate path for unverified track {track_id}")
                        continue
                    
                    # List all crop files in this track's folder
                    try:
                        crop_files = self.storage.list_blobs(prefix=unverified_prefix)
                        crops_to_move.extend(crop_files)
                        logger.debug(f"Found {len(crop_files)} crops in track {track_id}")
                    except Exception as e:
                        logger.error(f"Failed to list crops for track {track_id}: {e}")
                        failed_count += 1
                        continue
                
                # Move all collected crops to the verified_tracks/{group_id}/ folder
                verified_prefix = self.path_manager.get_path(
                    "verified_tracks",
                    video_id=self.current_process_folder,
                    track_id=group_id
                )
                
                if verified_prefix is None:
                    logger.error(f"Could not generate verified path for group {group_id}")
                    failed_count += len(crops_to_move)
                    continue
                
                for crop_blob in crops_to_move:
                    try:
                        # Construct the new path by replacing the unverified prefix with verified
                        # crop_blob.name is the full GCS path like "process/video1/unverified_tracks/123/crop_1_100.jpg"
                        # We need to change it to "process/video1/verified_tracks/456/crop_1_100.jpg"
                        
                        # Extract the filename from the original path
                        filename = crop_blob.name.split('/')[-1]  # e.g., "crop_1_100.jpg"
                        
                        # Construct new verified path
                        verified_path = f"{verified_prefix}{filename}"
                        
                        # Copy the blob to the new location
                        if self.storage.copy_blob(crop_blob.name, verified_path):
                            # Delete the original
                            if self.storage.delete_blob(crop_blob.name):
                                moved_count += 1
                                logger.debug(f"Moved crop {filename} to group {group_id}")
                            else:
                                logger.warning(f"Failed to delete original crop {crop_blob.name} after copying")
                                failed_count += 1
                        else:
                            logger.error(f"Failed to copy crop {crop_blob.name} to {verified_path}")
                            failed_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error moving crop {crop_blob.name}: {e}")
                        failed_count += 1
            
            logger.info(f"Crop migration complete: {moved_count} moved, {failed_count} failed")
            
            if failed_count > 0:
                logger.warning(f"Crop migration had {failed_count} failures")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during crop migration: {e}")
            return False
