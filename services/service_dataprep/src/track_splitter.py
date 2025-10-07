"""
Track splitting functionality for the LaxAI dataprep service.

This module provides functionality to split incorrectly merged tracks
at specific frame boundaries.
"""

import logging
from typing import Optional, Set
import numpy as np

from shared_libs.common.google_storage import GCSPaths, get_storage
from .stitcher import TrackStitcher, EdgeType

logger = logging.getLogger(__name__)


class TrackSplitter:
    """
    Handles splitting of tracks that were incorrectly merged by the tracker.

    This class provides methods to split a track into two parts at a specific
    frame boundary, updating all relevant data structures and moving crops.
    """

    def __init__(self, stitcher: TrackStitcher, path_manager: GCSPaths, storage, video_id: str, tenant_id: str):
        """
        Initialize the TrackSplitter.

        Args:
            stitcher: The TrackStitcher instance
            path_manager: GCS path manager
            storage: GCS storage client
            video_id: Current video ID
            tenant_id: Tenant identifier
        """
        self.stitcher = stitcher
        self.path_manager = path_manager
        self.storage = storage
        self.video_id = video_id
        self.tenant_id = tenant_id

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
        try:
            # Parse the crop image name to extract the frame number
            split_frame = self._parse_crop_frame(crop_image_name)
            if split_frame is None:
                logger.error(f"Could not parse frame number from crop image name: {crop_image_name}")
                return False

            logger.info(f"Splitting track {track_id} at frame {split_frame}")

            # Get all frames for this track
            track_frames = self.stitcher._track_to_frames.get(track_id)
            if track_frames is None:
                logger.error(f"Track {track_id} not found in stitcher")
                return False

            # Split frames at the boundary - crop at split_frame starts the new track
            first_part_frames = track_frames[track_frames < split_frame]
            second_part_frames = track_frames[track_frames >= split_frame]

            if len(second_part_frames) == 0:
                logger.error(f"No frames found at or after split frame {split_frame} for track {track_id}")
                return False

            # Generate new track ID
            new_track_id = self._generate_new_track_id()
            logger.info(f"Generated new track ID: {new_track_id}")

            # Update detections object
            if not self._update_detections_for_split(track_id, new_track_id, split_frame):
                logger.error("Failed to update detections object")
                return False

            # Update stitcher data structures
            if not self._update_stitcher_for_split(track_id, new_track_id, first_part_frames, second_part_frames):
                logger.error("Failed to update stitcher data structures")
                return False

            # Move crops to new track folder
            if not self._move_crops_for_split(track_id, new_track_id, split_frame):
                logger.error("Failed to move crops")
                return False

            logger.info(f"Successfully split track {track_id} into {track_id} (frames < {split_frame}) and {new_track_id} (frames >= {split_frame})")
            return True

        except Exception as e:
            logger.error(f"Failed to split track {track_id}: {e}")
            return False

    def _parse_crop_frame(self, crop_image_name: str) -> Optional[int]:
        """
        Parse the frame number from a crop image name.

        Args:
            crop_image_name: Crop filename (e.g., "crop_1_100.jpg" or "crop_960.jpg")

        Returns:
            Frame number or None if parsing fails
        """
        try:
            # Expected formats: 
            # - crop_X_Y.jpg where Y is the frame number
            # - crop_Z.jpg where Z is the frame number
            # Split by '_' and '.' to extract parts
            parts = crop_image_name.replace('.jpg', '').split('_')
            if parts[0] == 'crop':
                if len(parts) == 2:
                    # Format: crop_Z.jpg
                    return int(parts[1])
                elif len(parts) >= 3:
                    # Format: crop_X_Y.jpg (take the last part as frame number)
                    return int(parts[-1])
                else:
                    logger.error(f"Unexpected crop name format: {crop_image_name}")
                    return None
            else:
                logger.error(f"Crop name doesn't start with 'crop': {crop_image_name}")
                return None
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse frame from crop name {crop_image_name}: {e}")
            return None

    def _generate_new_track_id(self) -> int:
        """
        Generate a new unused track ID.

        Returns:
            New track ID that doesn't conflict with existing ones
        """
        # Get all existing track IDs
        existing_ids = set(self.stitcher._track_to_frames.keys())

        # Find the maximum and add 1
        if existing_ids:
            return max(existing_ids) + 1
        else:
            return 1

    def _update_detections_for_split(self, old_track_id: int, new_track_id: int, split_frame: int) -> bool:
        """
        Update the detections object to split the track.

        Args:
            old_track_id: Original track ID
            new_track_id: New track ID for the second part
            split_frame: Frame where the split occurs

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure detections is available
            assert self.stitcher.detections is not None, "Stitcher detections cannot be None"
            
            # Get the detections data
            frame_indices = self.stitcher.detections.data['frame_index']
            tracker_ids = self.stitcher.detections.tracker_id

            # Convert to numpy arrays if they aren't already
            frame_indices = np.asarray(frame_indices)
            tracker_ids = np.asarray(tracker_ids)

            # Find indices where tracker_id matches and frame > split_frame
            mask = (tracker_ids == old_track_id) & (frame_indices >= split_frame)

            # Update tracker_ids for the second part
            self.stitcher.detections.tracker_id[mask] = new_track_id  # type: ignore

            logger.info(f"Updated {mask.sum()} detections from track {old_track_id} to {new_track_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update detections: {e}")
            return False

    def _update_stitcher_for_split(self, old_track_id: int, new_track_id: int,
                                   first_part_frames: np.ndarray, second_part_frames: np.ndarray) -> bool:
        """
        Update stitcher data structures after splitting a track.

        Args:
            old_track_id: Original track ID
            new_track_id: New track ID
            first_part_frames: Frames for the first part (numpy array)
            second_part_frames: Frames for the second part (numpy array)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Update _track_to_frames
            self.stitcher._track_to_frames[old_track_id] = first_part_frames
            self.stitcher._track_to_frames[new_track_id] = second_part_frames

            # Update track_graph - add the new track node
            self.stitcher.track_graph.add_node(new_track_id)

            # Re-populate temporal conflicts for the new track
            self._repopulate_temporal_conflicts_for_track(new_track_id)

            # Update player_groups - the old track stays in its group, new track gets its own group
            self.stitcher.player_groups[new_track_id] = {new_track_id}

            logger.info(f"Updated stitcher data structures for split: {old_track_id} -> {old_track_id}, {new_track_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update stitcher: {e}")
            return False

    def _repopulate_temporal_conflicts_for_track(self, track_id: int):
        """
        Re-populate temporal conflicts for a newly added track.

        Args:
            track_id: The new track ID
        """
        for existing_track_id in self.stitcher.track_graph.nodes():
            if existing_track_id != track_id:
                if self.stitcher._tracks_overlap_in_time(track_id, existing_track_id):
                    self.stitcher.track_graph.add_edge(track_id, existing_track_id,
                                                     relationship=EdgeType.TEMPORAL_CONFLICT)

    def _move_crops_for_split(self, old_track_id: int, new_track_id: int, split_frame: int) -> bool:
        """
        Move crops from the old track folder to the new track folder for frames >= split_frame.

        Args:
            old_track_id: Original track ID
            new_track_id: New track ID
            split_frame: Frame where the split occurs (this frame starts the new track)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get paths for old and new track folders
            old_track_prefix = self.path_manager.get_path(
                "unverified_tracks",
                video_id=self.video_id,
                track_id=old_track_id
            )

            new_track_prefix = self.path_manager.get_path(
                "unverified_tracks",
                video_id=self.video_id,
                track_id=new_track_id
            )

            if old_track_prefix is None or new_track_prefix is None:
                logger.error("Failed to generate track paths")
                return False

            # List all crops in the old track folder
            crop_blobs = self.storage.list_blobs(prefix=old_track_prefix)

            moved_count = 0
            for crop_blob in crop_blobs:
                try:
                    # Parse frame number from crop name
                    filename = crop_blob.split('/')[-1]
                    crop_frame = self._parse_crop_frame(filename)

                    if crop_frame is None:
                        logger.warning(f"Could not parse frame from crop {filename}, skipping")
                        continue

                    # Only move crops with frame >= split_frame (split_frame starts the new track)
                    if crop_frame >= split_frame:
                        # Construct new path
                        new_path = f"{new_track_prefix}{filename}"

                        # Copy to new location
                        if self.storage.copy_blob(crop_blob, new_path):
                            # Delete from old location
                            if self.storage.delete_blob(crop_blob):
                                moved_count += 1
                                logger.debug(f"Moved crop {filename} to new track {new_track_id}")
                            else:
                                logger.warning(f"Failed to delete original crop {crop_blob} after copying")
                        else:
                            logger.error(f"Failed to copy crop {crop_blob} to {new_path}")

                except Exception as e:
                    logger.error(f"Error moving crop {crop_blob}: {e}")

            logger.info(f"Moved {moved_count} crops from track {old_track_id} to {new_track_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to move crops: {e}")
            return False