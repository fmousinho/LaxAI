"""
Cloud Run Function to Create Dataset from Player Manager Data.

This HTTP-triggered Cloud Function processes player data from a completed
dataprep session and creates a structured dataset for training by grouping
player images into train/validation splits and organizing verified track crops
into connected component folders.

Environment Variables:
    - GOOGLE_CLOUD_PROJECT: Automatically populated by GCP with the project ID.
"""

import logging
from typing import Dict, Any
import json
import random
from fastapi import FastAPI, HTTPException
import networkx as nx

from shared_libs.common.google_storage import GCSPaths, get_storage
from shared_libs.config import logging_config  # noqa: F401
from shared_libs.utils.id_generator import create_dataset_id

logger = logging.getLogger(__name__)

app = FastAPI(title="Create Dataset from Player Manager Data", version="1.0.0")

TRAINING_RATIO = 0.8

@app.post("/create-dataset-from-player-mgr")
async def create_dataset(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Create a training dataset from player manager data.

    Expects JSON: {"tenant_id": "tenant123", "video_id": "video_abc"}
    """
    try:
        tenant_id = request.get("tenant_id")
        video_id = request.get("video_id")

        if not tenant_id or not video_id:
            raise HTTPException(status_code=400, detail="Missing tenant_id or video_id")

        logger.info(f"Processing dataset creation for tenant {tenant_id}, video {video_id}")

        # Initialize path manager and storage
        path_manager = GCSPaths()
        storage = get_storage(tenant_id)

        # Load player data dictionary
        player_data_path = path_manager.get_path("players_data_path", video_id=video_id)
        if not player_data_path:
            raise HTTPException(status_code=404, detail="Player data path not found")
        if not storage.blob_exists(player_data_path):
            raise HTTPException(status_code=404, detail="Player data file not found")
        player_data_content = storage.download_as_string(player_data_path)
        if player_data_content is None:
            raise HTTPException(status_code=500, detail="Failed to download player data file")
        player_json = json.loads(player_data_content)
        # Extract track_to_player mapping as Python dictionary
        players_data = player_json.get("players", {})
        if not players_data or len(players_data) == 0:
            raise HTTPException(status_code=404, detail="No players mapping found")
        logger.info(f"Loaded players mapping with {len(players_data)} entries")

        # Create dataset folder
        dataset_id = create_dataset_id(video_id)

        train_img_count = 0
        val_img_count = 0

        for player_data in players_data:

            #Get list of images for the player
            images = set()
            player_id = player_data.get("player_id")
            player_tracks = player_data.get("tracker_ids", [])
            logger.info(f"Processing player {player_id} with {len(player_tracks)} tracks")
            for track in player_tracks:
                path = path_manager.get_path("unverified_tracks", video_id=video_id, track_id=track)
                if path is None or not storage.blob_exists(path):
                    logger.warning(f"Track path {path} does not exist for track {track}")
                    continue
                track_images = storage.list_blobs(prefix=path)
                images.update(track_images)
            logger.info(f"Player {player_id} has {len(images)} images across tracks")

            #Create train and val splits
            if len(images) < 3:
                continue  # Skip players with less than 3 images
            elif len(images) < 15:
                train_images = images  # Not enough players for a val dataset (min of 3)
                val_images = None
            else:
                train_images = []
                val_images = []
                images_list = list(images)
                random.shuffle(images_list)
                num_train = int(len(images_list) * TRAINING_RATIO)
                train_images.extend(images_list[:num_train])
                val_images.extend(images_list[num_train:])


            # Create player dataset folders and copy images
            train_player_folder = path_manager.get_path("train_player_folder",
                                                        dataset_id=dataset_id,
                                                        orig_crop_id=player_id)
            for img_path in train_images:
                img_filename = img_path.split('/')[-1]
                storage.copy_blob(img_path, f"{train_player_folder}{img_filename}")
                train_img_count += 1
            if val_images:
                val_player_folder = path_manager.get_path("val_player_folder",
                                                          dataset_id=dataset_id,
                                                          orig_crop_id=player_id)
                for img_path in val_images:
                    img_filename = img_path.split('/')[-1]
                    storage.copy_blob(img_path, f"{val_player_folder}{img_filename}")
                    val_img_count += 1

        logger.info(f"Created dataset {dataset_id} with {train_img_count} training images and {val_img_count} validation images.")
        return {"success": True, "dataset_id": dataset_id, "train": train_img_count, "val": val_img_count}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)