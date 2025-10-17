"""
DataPrep Service SDK Manager

This module provides a Python SDK for the LaxAI dataprep service, exposing
functionality for track stitching verification workflows.
"""

import logging
import os
import signal
import threading
import weakref
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from shared_libs.common.google_storage import GCSPaths, get_storage
from shared_libs.common.detection_utils import json_to_detections
from ..stitcher import TrackStitcher  # type: ignore
from ..track_splitter import TrackSplitter
from ..pair_tracker import VerificationPairTracker

logger = logging.getLogger(__name__)


class DataPrepManager:
    """
    Manager class for dataprep service operations.

    Provides methods to list process folders, start verification sessions,
    get images for verification, record user responses, and suspend/resume sessions.
    """

    _signal_handler_installed = False
    _instances = weakref.WeakSet()
    _original_signal_handlers: Dict[int, Any] = {}

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
        self.current_video_id: Optional[str] = None

        # Verification pair management
        self._lock = threading.RLock()
        self._max_outstanding_pairs = int(os.getenv("DATAPREP_MAX_OUTSTANDING_PAIRS", "10"))
        self._pair_expiration_seconds = int(os.getenv("DATAPREP_PAIR_EXPIRATION_SECONDS", "600"))
        self._pair_tracker = VerificationPairTracker(
            max_outstanding_pairs=self._max_outstanding_pairs,
            pair_ttl_seconds=self._pair_expiration_seconds,
            release_callback=self._release_stitcher_pair,
        )

        # Autosave configuration
        self._autosave_interval_seconds = int(os.getenv("DATAPREP_AUTOSAVE_INTERVAL_SECONDS", "600"))
        self._autosave_thread: Optional[threading.Thread] = None
        self._autosave_stop_event: Optional[threading.Event] = None
        self._shutdown_in_progress = False

        # Track active instances for signal handling
        DataPrepManager._instances.add(self)
        self._ensure_signal_handlers_registered()

    @classmethod
    def _ensure_signal_handlers_registered(cls) -> None:
        if cls._signal_handler_installed:
            return

        def _handle_signal(signum: int, frame) -> None:  # type: ignore[override]
            logger.info("Received shutdown signal %s; scheduling dataprep graph persistence", signum)
            for manager in list(cls._instances):
                try:
                    manager._schedule_shutdown_save(signum)
                except Exception:
                    logger.exception("Failed to schedule shutdown save for tenant %s", getattr(manager, "tenant_id", "unknown"))

            previous = cls._original_signal_handlers.get(signum)
            if callable(previous) and previous is not _handle_signal:
                try:
                    previous(signum, frame)
                except Exception:
                    logger.exception("Error while delegating signal %s to original handler", signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                cls._original_signal_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _handle_signal)
            except (ValueError, OSError) as exc:
                logger.warning("Unable to register dataprep shutdown handler for signal %s: %s", sig, exc)

        cls._signal_handler_installed = True

    def _schedule_shutdown_save(self, signum: int) -> None:
        if self._shutdown_in_progress:
            return

        if self.stitcher is None or self.current_video_id is None:
            return

        self._shutdown_in_progress = True

        def _shutdown_worker() -> None:
            try:
                logger.info(
                    "Shutdown signal %s received; attempting to persist dataprep state for tenant %s",
                    signum,
                    self.tenant_id,
                )
                self._stop_autosave_loop()
                with self._lock:
                    if self.stitcher is None or self.current_video_id is None:
                        logger.info("No active session during shutdown for tenant %s", self.tenant_id)
                        return
                    success = self._save_graph_internal(reason="shutdown")
                    if success:
                        logger.info("Successfully saved dataprep graph during shutdown for tenant %s", self.tenant_id)
                    else:
                        logger.warning("Failed to save dataprep graph during shutdown for tenant %s", self.tenant_id)
            except Exception:
                logger.exception("Unexpected error while saving dataprep graph during shutdown for tenant %s", self.tenant_id)
            finally:
                self._shutdown_in_progress = False

        threading.Thread(
            target=_shutdown_worker,
            name=f"dataprep-shutdown-save-{self.tenant_id}",
            daemon=True,
        ).start()

    def _start_autosave_loop(self) -> None:
        if self._autosave_interval_seconds <= 0:
            logger.info("Autosave interval disabled (<=0); skipping background saves for tenant %s", self.tenant_id)
            return

        if self._autosave_thread and self._autosave_thread.is_alive():
            return

        self._autosave_stop_event = threading.Event()

        def _autosave_loop(stop_event: threading.Event) -> None:
            logger.info(
                "Autosave loop started for tenant %s with interval %ss",
                self.tenant_id,
                self._autosave_interval_seconds,
            )
            try:
                while not stop_event.wait(self._autosave_interval_seconds):
                    if self.stitcher is None or self.current_video_id is None:
                        continue
                    try:
                        with self._lock:
                            if self.stitcher is None or self.current_video_id is None:
                                continue
                            success = self._save_graph_internal(reason="autosave")
                            if success:
                                logger.info("Autosave completed for tenant %s", self.tenant_id)
                            else:
                                logger.warning("Autosave failed for tenant %s", self.tenant_id)
                    except Exception:
                        logger.exception("Autosave encountered an error for tenant %s", self.tenant_id)
            finally:
                logger.info("Autosave loop exiting for tenant %s", self.tenant_id)

        self._autosave_thread = threading.Thread(
            target=_autosave_loop,
            args=(self._autosave_stop_event,),
            name=f"dataprep-autosave-{self.tenant_id}",
            daemon=True,
        )
        self._autosave_thread.start()

    def _stop_autosave_loop(self) -> None:
        if self._autosave_stop_event is not None:
            self._autosave_stop_event.set()

        if self._autosave_thread and self._autosave_thread.is_alive():
            self._autosave_thread.join(timeout=5)

        self._autosave_thread = None
        self._autosave_stop_event = None

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

    def start_prep(self, video_id: str) -> bool:
        """
        Start a track stitching verification session for a video.

        Args:
            video_id: The video ID (can be full path, filename with .mp4, or filename without extension)

        Returns:
            True if successfully started, False otherwise
        """
        try:
            # Normalize the video_id by removing .mp4 extension if present
            if video_id.endswith('.mp4/'):
                video_id = video_id[:-5] + '/'  # Remove '.mp4/'
            elif video_id.endswith('.mp4'):
                video_id = video_id[:-4]  # Remove '.mp4'
            
            # Check if video file exists in the imported_video path
            imported_video_path = self.path_manager.get_path(
                "imported_video",
                video_id=video_id
            )
            if type(imported_video_path) is not str:
                logger.error(f"Invalid imported video path: {imported_video_path}")
                return False
            
            # List files in the imported_video directory to check for video file
            try:
                blobs = self.storage.list_blobs(prefix=imported_video_path)
                video_files = [blob for blob in blobs if blob.endswith('.mp4')]
                if not video_files:
                    logger.error(f"No video file found in {imported_video_path}. Expected a .mp4 file.")
                    return False
                elif len(video_files) > 1:
                    logger.warning(f"Multiple video files found in {imported_video_path}, using first: {video_files[0]}")
                video_file_path = video_files[0]
                logger.info(f"Found video file: {video_file_path}")
            except Exception as e:
                logger.error(f"Failed to check for video file in {imported_video_path}: {e}")
                return False
            
            # Get the detections.json path
            detections_path = self.path_manager.get_path(
                "detections_path",
                video_id=video_id
            )
            if type(detections_path) is not str:
                logger.error(f"Invalid detections path: {detections_path}")
                return False

            # Load detections from GCS
            logger.info(f"Attempting to download detections from: {detections_path}")
            detections_json_text = self.storage.download_as_string(detections_path)
            if detections_json_text is None:
                logger.error(f"Could not download detections from {detections_path}. "
                           f"Please ensure the video has been processed by the tracking service first. "
                           f"Check that the video_id '{video_id}' is correct.")
                return False
            
            import json
            detections_json = json.loads(detections_json_text)
            
            # Handle different JSON formats - extract detections list if wrapped in object
            if isinstance(detections_json, dict) and "detections" in detections_json:
                detections_list = detections_json["detections"]
            elif isinstance(detections_json, list):
                detections_list = detections_json
            else:
                logger.error(f"Invalid detections JSON format: {type(detections_json)}")
                return False
            
            # Load detections - the loader automatically handles different formats
            detections = json_to_detections(detections_list)
            logger.info("Successfully loaded detections")

            # Check for existing saved graph
            existing_graph = None
            saved_graph_path = self.path_manager.get_path(
                "saved_graph",
                video_id=video_id
            )
            
            if saved_graph_path is not None:
                try:
                    # Try to download the saved graph
                    graph_data = self.storage.download_as_string(saved_graph_path)
                    if graph_data is not None:
                        import networkx as nx
                        import io
                        import tempfile
                        import os
                        
                        # Create a temporary file to load the graph
                        with tempfile.NamedTemporaryFile(mode='w+', suffix='.gml', delete=False) as temp_file:
                            temp_file.write(graph_data)
                            temp_filepath = temp_file.name
                        
                        try:
                            # Load the graph from the temporary file
                            existing_graph = nx.read_gml(temp_filepath)
                            logger.info(f"Found existing saved graph, will resume from {saved_graph_path}")
                        finally:
                            # Clean up the temporary file
                            os.unlink(temp_filepath)
                    else:
                        logger.info("No existing saved graph found, starting fresh")
                except Exception as e:
                    logger.warning(f"Could not load existing saved graph: {e}, starting fresh")
            
            logger.info("Starting fresh or resuming from saved graph")

            # Clear any outstanding verification bookkeeping from a previous session
            self._pair_tracker.reset()

            # Ensure any previous autosave loop is stopped before creating a new stitcher
            self._stop_autosave_loop()

            # Create the stitcher
            if isinstance(detections, tuple):
                detections = detections[0]
            self.stitcher = TrackStitcher(detections=detections, existing_graph=existing_graph, storage=self.storage, video_id=video_id, path_manager=self.path_manager)
            self.current_video_id = video_id

            # Begin periodic autosaves for the active session
            self._start_autosave_loop()

            logger.info(f"Started prep session for video: {video_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start prep for {video_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Verification pair management helpers
    # ------------------------------------------------------------------

    def _release_stitcher_pair(self, group1_id: int, group2_id: int) -> None:
        if self.stitcher is not None:
            self.stitcher.release_in_progress_pair(group1_id, group2_id)

    def _cleanup_expired_pairs(self) -> List[str]:
        """Remove any outstanding pairs that have exceeded their TTL."""
        expired_ids = self._pair_tracker.cleanup_expired()
        if expired_ids:
            logger.info(
                "Expired %d verification pairs for tenant %s: %s",
                len(expired_ids),
                self.tenant_id,
                expired_ids,
            )
        return expired_ids

    def get_images_for_verification(self) -> Dict[str, Any]:
        """
        Get the next pair of track groups for verification.

        Returns:
            Dictionary containing verification pair metadata or status messages.
        """
        if self.stitcher is None:
            return {"status": "error", "message": "No active stitcher session"}

        if self.current_video_id is None:
            return {"status": "error", "message": "No current video"}

        with self._lock:
            self._cleanup_expired_pairs()

            if not self._pair_tracker.has_capacity():
                logger.info(
                    "Outstanding verification pair limit reached (%d) for tenant %s",
                    self._pair_tracker.max_outstanding_pairs,
                    self.tenant_id,
                )
                return {
                    "status": "capacity_exceeded",
                    "message": "Too many verification pairs are pending. Please retry later.",
                    "outstanding_pair_ids": self._pair_tracker.outstanding_pair_ids(),
                    "max_outstanding_pairs": self._pair_tracker.max_outstanding_pairs,
                }

            try:
                result = self.stitcher.get_pair_for_verification()

                if result["status"] == "pending_verification":
                    group1_id = result["group1_id"]
                    group2_id = result["group2_id"]
                    mode = result.get("mode", self.stitcher.verification_mode)
                    issued_at = datetime.now(timezone.utc)

                    # Gather auxiliary metadata prior to registering the pair so we can roll back on failure
                    group1_prefixes = self._get_group_track_prefixes(group1_id)
                    group2_prefixes = self._get_group_track_prefixes(group2_id)
                    progress_info = self.stitcher.get_verification_progress()

                    self.stitcher.mark_pair_in_progress(group1_id, group2_id)

                    try:
                        pair_entry = self._pair_tracker.register_pair(
                            group1_id=group1_id,
                            group2_id=group2_id,
                            mode=mode,
                            issued_at=issued_at,
                            ttl_seconds=self._pair_expiration_seconds,
                        )
                    except Exception:
                        self.stitcher.release_in_progress_pair(group1_id, group2_id)
                        raise
                    logger.info(
                        "Issuing verification pair %s (groups %d, %d) for tenant %s",
                        pair_entry.pair_id,
                        group1_id,
                        group2_id,
                        self.tenant_id,
                    )

                    try:
                        response: Dict[str, Any] = {
                            "status": "pending_verification",
                            "pair_id": pair_entry.pair_id,
                            "group1_id": group1_id,
                            "group2_id": group2_id,
                            "mode": mode,
                            "issued_at": pair_entry.issued_at,
                            "expires_at": pair_entry.expires_at,
                            "group1_prefixes": group1_prefixes,
                            "group2_prefixes": group2_prefixes,
                            "total_pairs": progress_info["total_possible_pairs"],
                            "verified_pairs": progress_info["verified_pairs"],
                            "remaining_pairs": progress_info["remaining_pairs"],
                            "outstanding_pair_ids": self._pair_tracker.outstanding_pair_ids(),
                            "max_outstanding_pairs": self._pair_tracker.max_outstanding_pairs,
                        }
                        return response
                    except Exception:
                        # Something failed after registering the pair – roll back bookkeeping and re-raise
                        self._pair_tracker.expire_pair(pair_entry.pair_id, "failed_during_pair_issue")
                        self.stitcher.release_in_progress_pair(group1_id, group2_id)
                        raise

                # Non-pending status: persist graph when verification has concluded
                if result["status"] in {"complete", "second_pass_ready"}:
                    if not self.save_graph():
                        logger.warning("Failed to save graph after completing verification")

                result.setdefault("outstanding_pair_ids", self._pair_tracker.outstanding_pair_ids())
                result.setdefault("max_outstanding_pairs", self._pair_tracker.max_outstanding_pairs)
                return result

            except Exception as e:
                logger.error(f"Failed to get images for verification: {e}")
                return {"status": "error", "message": str(e)}

    def record_response(self, pair_id: str, decision: str) -> Dict[str, Any]:
        """Record a user response for a specific verification pair."""
        logger.info(
            f"[MANAGER] record_response called: pair_id={pair_id!r} (type={type(pair_id).__name__}), decision={decision}"
        )

        if self.stitcher is None:
            logger.error("No active stitcher session")
            return {
                "success": False,
                "message": "No active stitcher session",
                "pair_id": pair_id,
                "pair_status": "error",
                "outstanding_pair_ids": self._pair_tracker.outstanding_pair_ids(),
                "max_outstanding_pairs": self._pair_tracker.max_outstanding_pairs,
            }

        with self._lock:
            self._cleanup_expired_pairs()

            logger.info(
                f"[MANAGER] Looking up pair_id={pair_id!r}. Current outstanding pairs: {self._pair_tracker.outstanding_pair_ids()}"
            )
            pair = self._pair_tracker.get_pair(pair_id)
            if pair is None:
                logger.warning(
                    f"[MANAGER] Attempt to record response for unknown pair {pair_id!r}. "
                    f"Outstanding pairs: {self._pair_tracker.outstanding_pair_ids()}"
                )
                return {
                    "success": False,
                    "message": f"Pair {pair_id} is not pending. Please request a new verification pair.",
                    "pair_id": pair_id,
                    "pair_status": "unknown",
                    "outstanding_pair_ids": self._pair_tracker.outstanding_pair_ids(),
                    "max_outstanding_pairs": self._pair_tracker.max_outstanding_pairs,
                }

            now = datetime.now(timezone.utc)
            if pair.expires_at <= now:
                self._pair_tracker.expire_pair(pair_id, "expired before response")
                logger.info(
                    "Verification pair %s expired before response for tenant %s",
                    pair_id,
                    self.tenant_id,
                )
                return {
                    "success": False,
                    "message": f"Pair {pair_id} expired. Please request a new verification pair.",
                    "pair_id": pair_id,
                    "pair_status": "expired",
                    "outstanding_pair_ids": self._pair_tracker.outstanding_pair_ids(),
                    "max_outstanding_pairs": self._pair_tracker.max_outstanding_pairs,
                }

            # Check if groups still exist (not merged away)
            if pair.group1_id not in self.stitcher.player_groups or pair.group2_id not in self.stitcher.player_groups:
                self._pair_tracker.expire_pair(pair_id, "invalidated due to merge")
                logger.info(
                    "Verification pair %s invalidated due to merge for tenant %s",
                    pair_id,
                    self.tenant_id,
                )
                return {
                    "success": False,
                    "message": f"Pair {pair_id} invalidated due to group merge. Please request a new verification pair.",
                    "pair_id": pair_id,
                    "pair_status": "invalidated",
                    "outstanding_pair_ids": self._pair_tracker.outstanding_pair_ids(),
                    "max_outstanding_pairs": self._pair_tracker.max_outstanding_pairs,
                }

            try:
                self.stitcher.respond_to_pair(pair.group1_id, pair.group2_id, decision, mode=pair.mode)
            except ValueError as exc:
                logger.error("Invalid decision %s for pair %s: %s", decision, pair_id, exc)
                self._pair_tracker.expire_pair(pair_id, "invalid_decision")
                return {
                    "success": False,
                    "message": str(exc),
                    "pair_id": pair_id,
                    "pair_status": "error",
                    "outstanding_pair_ids": self._pair_tracker.outstanding_pair_ids(),
                    "max_outstanding_pairs": self._pair_tracker.max_outstanding_pairs,
                }
            except Exception as exc:
                logger.error(f"Failed to record response '{decision}' for pair {pair_id}: {exc}")
                self._pair_tracker.expire_pair(pair_id, "error_during_response")
                return {
                    "success": False,
                    "message": f"Failed to record response '{decision}': {exc}",
                    "pair_id": pair_id,
                    "pair_status": "error",
                    "outstanding_pair_ids": self._pair_tracker.outstanding_pair_ids(),
                    "max_outstanding_pairs": self._pair_tracker.max_outstanding_pairs,
                }

            logger.info(
                "About to complete pair %s (tenant %s). Outstanding before completion: %d, pairs: %s",
                pair_id,
                self.tenant_id,
                self._pair_tracker.active_count,
                self._pair_tracker.outstanding_pair_ids(),
            )
            
            completed_pair = self._pair_tracker.complete_pair(pair_id, "completed")
            if completed_pair is None:
                logger.warning(
                    "Completed pair %s was not found in tracker for tenant %s; outstanding now %s",
                    pair_id,
                    self.tenant_id,
                    self._pair_tracker.outstanding_pair_ids(),
                )

            logger.info(
                "Recorded decision %s for pair %s (tenant %s). Outstanding pairs after: %d, pairs: %s",
                decision,
                pair_id,
                self.tenant_id,
                self._pair_tracker.active_count,
                self._pair_tracker.outstanding_pair_ids(),
            )
            return {
                "success": True,
                "message": f"Recorded decision: {decision}",
                "pair_id": pair_id,
                "pair_status": "completed",
                "outstanding_pair_ids": self._pair_tracker.outstanding_pair_ids(),
                "max_outstanding_pairs": self._pair_tracker.max_outstanding_pairs,
            }

    def _get_group_track_prefixes(self, group_id: int) -> List[str]:
        """
        Get GCS prefixes for all tracks in a group.

        Args:
            group_id: The group ID

        Returns:
            List of GCS prefixes for the tracks in the group
        """
        if self.stitcher is None or self.current_video_id is None:
            return []

        # Get the tracks for this group
        group_tracks = self.stitcher.player_groups.get(group_id, [])

        prefixes = []
        for track_id in group_tracks:
            # Get the unverified_tracks path for this track
            prefix = self.path_manager.get_path(
                "unverified_tracks",
                video_id=self.current_video_id,
                track_id=track_id
            )
            prefix = f"{self.storage.bucket_name}/{self.tenant_id}/{prefix}"
            prefixes.append(prefix)

        return prefixes

    def save_graph(self) -> bool:
        """
        Save the current stitcher graph state to GCS.

        Returns:
            True if the graph was successfully saved, False otherwise
        """
        with self._lock:
            return self._save_graph_internal(reason="manual")

    def _save_graph_internal(self, reason: str) -> bool:
        if self.stitcher is None:
            logger.debug("Skipping %s save attempt: no active stitcher session", reason)
            return False

        if self.current_video_id is None:
            logger.debug("Skipping %s save attempt: no current video set", reason)
            return False

        import tempfile
        import os

        try:
            # Create a temporary file for the graph
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.gml', delete=False) as temp_file:
                temp_filepath = temp_file.name

            # Save the graph to the temporary file
            success = self.stitcher.save_graph(temp_filepath, format="gml")
            if not success:
                logger.error("Failed to persist graph to temporary file during %s save", reason)
                os.unlink(temp_filepath)
                return False

            # Get the GCS path for the saved graph
            gcs_path = self.path_manager.get_path(
                "saved_graph",
                video_id=self.current_video_id
            )

            if gcs_path is None:
                logger.error("Failed to generate GCS path for saved graph during %s save", reason)
                os.unlink(temp_filepath)
                return False

            # Upload the file to GCS
            upload_success = self.storage.upload_from_file(gcs_path, temp_filepath)

            # Clean up the temporary file
            os.unlink(temp_filepath)

            if upload_success:
                logger.info("Successfully saved graph to %s via %s save", gcs_path, reason)
                return True
            else:
                logger.error("Failed to upload graph to GCS during %s save", reason)
                return False

        except Exception as e:
            logger.error("Error saving graph during %s save: %s", reason, e)
            return False

    def save_graph_image(self) -> tuple[bool, Optional[str]]:
        """
        Generate and save a visual representation of the track relationship graph to GCS.

        Returns:
            Tuple of (success: bool, image_url: Optional[str]) where image_url is the full GCS path
            with gs:// prefix if successful, None otherwise
        """
        if self.stitcher is None:
            logger.error("No active stitcher session to visualize")
            return False, None

        if self.current_video_id is None:
            logger.error("No current video set")
            return False, None

        import tempfile
        import os

        try:
            # Generate the visualization image
            image = self.stitcher.visualize_graph()
            if image is None:
                logger.error("Failed to generate graph visualization")
                return False, None

            # Create a temporary file for the image
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.jpg', delete=False) as temp_file:
                temp_filepath = temp_file.name

            # Save the PIL image to the temporary file
            image.save(temp_filepath, format='JPEG', quality=95)
            image.close()  # Free memory

            # Get the GCS path for the graph image
            gcs_path = self.path_manager.get_path(
                "track_graph_image",
                video_id=self.current_video_id
            )

            if gcs_path is None:
                logger.error("Failed to generate GCS path for graph image")
                os.unlink(temp_filepath)
                return False, None

            # Upload the file to GCS
            upload_success = self.storage.upload_from_file(gcs_path, temp_filepath)

            # Clean up the temporary file
            os.unlink(temp_filepath)

            if upload_success:
                # Construct the full GCS URL with gs:// prefix
                full_gcs_url = f"gs://{self.storage.bucket_name}/{gcs_path}"
                logger.info(f"Successfully saved graph visualization to {full_gcs_url}")
                return True, full_gcs_url
            else:
                logger.error("Failed to upload graph image to GCS")
                return False, None

        except Exception as e:
            logger.error(f"Error saving graph image: {e}")
            return False, None

    def suspend_prep(self) -> bool:
        """
        Save the current stitcher graph state to GCS for later resumption.

        Returns:
            True if the graph was successfully saved, False otherwise
        """
        success = self.save_graph()
        if success:
            self._stop_autosave_loop()
            # Clear the session state to allow starting a new session
            self.stitcher = None
            self.current_video_id = None
            logger.info("Session suspended and cleared")
        return success

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

        if self.current_video_id is None:
            logger.error("No current video set")
            return False

        try:
            logger.info(f"Starting crop migration for video {self.current_video_id}")
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
                        video_id=self.current_video_id,
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
                    video_id=self.current_video_id,
                    track_id=group_id
                )
                
                if verified_prefix is None:
                    logger.error(f"Could not generate verified path for group {group_id}")
                    failed_count += len(crops_to_move)
                    continue
                
                for crop_blob in crops_to_move:
                    try:
                        # Construct the new path by replacing the unverified prefix with verified
                        # crop_blob is the full GCS path like "process/video1/unverified_tracks/123/crop_1_100.jpg"
                        # We need to change it to "process/video1/verified_tracks/456/crop_1_100.jpg"
                        
                        # Extract the filename from the original path
                        filename = crop_blob.split('/')[-1]  # e.g., "crop_1_100.jpg"
                        
                        # Construct new verified path
                        verified_path = f"{verified_prefix}{filename}"
                        
                        # Copy the blob to the new location
                        if self.storage.copy_blob(crop_blob, verified_path):
                            # Delete the original
                            if self.storage.delete_blob(crop_blob):
                                moved_count += 1
                                logger.debug(f"Moved crop {filename} to group {group_id}")
                            else:
                                logger.warning(f"Failed to delete original crop {crop_blob} after copying")
                                failed_count += 1
                        else:
                            logger.error(f"Failed to copy crop {crop_blob} to {verified_path}")
                            failed_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error moving crop {crop_blob}: {e}")
                        failed_count += 1
            
            logger.info(f"Crop migration complete: {moved_count} moved, {failed_count} failed")
            
            if failed_count > 0:
                logger.warning(f"Crop migration had {failed_count} failures")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during crop migration: {e}")
            return False

    def split_track_at_frame(self, track_id: int, crop_image_name: str) -> bool:
        """
        Split a track into two parts at the specified crop frame.

        This function corrects cases where the tracker incorrectly grouped two players
        in the same track. It splits the track at the frame where the player switch occurs,
        keeping the original track_id for frames up to and including the split frame,
        and creating a new track_id for frames after the split.

        Args:
            track_id: The original track ID to split
            crop_image_name: Name of the crop image where the player switch occurs
                           (e.g., "crop_1_100.jpg" where 100 is the frame number)

        Returns:
            True if the track was successfully split, False otherwise
        """
        if self.stitcher is None:
            logger.error("No active stitcher session")
            return False

        if self.current_video_id is None:
            logger.error("No current video set")
            return False

        # Create track splitter and delegate the work
        splitter = TrackSplitter(
            stitcher=self.stitcher,
            path_manager=self.path_manager,
            storage=self.storage,
            video_id=self.current_video_id,
            tenant_id=self.tenant_id
        )

        return splitter.split_track_at_frame(track_id, crop_image_name)
