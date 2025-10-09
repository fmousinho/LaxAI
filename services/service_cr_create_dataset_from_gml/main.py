"""
Cloud Run Function to Create Dataset from GML Graph.

This HTTP-triggered Cloud Function processes a GML graph file from a completed
dataprep session and creates a structured dataset for training by grouping
verified track crops into connected component folders.

Environment Variables:
    - GOOGLE_CLOUD_PROJECT: Automatically populated by GCP with the project ID.
"""

import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
import networkx as nx

from shared_libs.common.google_storage import GCSPaths, get_storage
from shared_libs.config import logging_config  # noqa: F401
from shared_libs.utils.id_generator import create_dataset_id

logger = logging.getLogger(__name__)

app = FastAPI(title="Create Dataset from GML", version="1.0.0")


@app.post("/create-dataset-from-gml")
async def create_dataset_from_gml(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Create a training dataset from a GML graph file.

    Expects JSON: {"tenant_id": "tenant123", "video_name": "video_abc"}
    """
    try:
        tenant_id = request.get("tenant_id")
        video_name = request.get("video_name")

        if not tenant_id or not video_name:
            raise HTTPException(status_code=400, detail="Missing tenant_id or video_name")

        logger.info(f"Processing dataset creation for tenant {tenant_id}, video {video_name}")

        # Initialize path manager and storage
        path_manager = GCSPaths()
        storage = get_storage(tenant_id)

        # Get paths
        process_root = path_manager.get_path("process_root", tenant_id=tenant_id)
        gml_path = f"{process_root}{video_name}/stitcher_graph.gml"
        unverified_tracks_path = f"{process_root}{video_name}/unverified_tracks/"

        # Validate required files/folders exist
        if not storage.blob_exists(gml_path):
            raise HTTPException(status_code=400, detail="GML file not found in process folder")

        if not storage.list_blobs(prefix=unverified_tracks_path):
            raise HTTPException(status_code=400, detail="Unverified tracks folder not found")

        # Load and parse GML graph
        gml_content = storage.download_as_string(gml_path)
        G = nx.parse_gml(gml_content)

        # Identify verified tracks (those with edges)
        verified_tracks = set()
        for u, v in G.edges():
            verified_tracks.add(u)
            verified_tracks.add(v)

        # Get connected components
        components = list(nx.connected_components(G))

        # Create dataset
        dataset_id = create_dataset_id(video_name)
        dataset_path = path_manager.get_path("dataset_folder", dataset_id=dataset_id, tenant_id=tenant_id)

        component_num = 1
        for component in components:
            # Skip components with no verified tracks
            if not any(track in verified_tracks for track in component):
                continue

            component_folder = f"{dataset_path}{component_num}/"

            # Copy crops from verified tracks in this component
            for track in component:
                if track in verified_tracks:
                    track_folder = f"{unverified_tracks_path}{track}/"
                    crops = storage.list_blobs(prefix=track_folder)
                    for crop_name in crops:
                        dest_path = crop_name.replace(track_folder, component_folder)
                        storage.copy_blob(crop_name, dest_path)

            component_num += 1

        logger.info(f"Dataset {dataset_id} created successfully with {component_num-1} components")
        return {"success": True, "dataset_id": dataset_id, "components": component_num-1}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)