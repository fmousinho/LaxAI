# Create Dataset from GML Cloud Function

This Cloud Run service processes a GML graph file from a completed dataprep session and creates a structured training dataset by grouping verified track crops into connected component folders.

## Endpoint

`POST /create-dataset-from-gml`

### Request Body
```json
{
  "tenant_id": "tenant123",
  "video_name": "video_abc"
}
```

### Response
```json
{
  "success": true,
  "dataset_id": "video_abc",
  "components": 3
}
```

## Functionality

1. Validates that the process folder contains a `stitcher_graph.gml` file and `unverified_tracks/` folder.
2. Loads the GML graph and identifies connected components.
3. Filters for tracks that have verification relationships (edges in the graph).
4. Creates a dataset folder with subfolders for each connected component (1/, 2/, etc.).
5. Copies crop images from verified tracks into their respective component folders.

## Deployment

Deploy using Cloud Build with the corresponding cloudbuild yaml file.