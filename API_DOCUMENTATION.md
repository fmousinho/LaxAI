# LaxAI Video Stitcher API

**Version:** 1.0.0

**Base URL:** https://your-service-url

## Endpoints

### POST /stitcher/video/load

**Summary:** Loads video for stitching analysis

**Description:** Start a track stitching verification session for a specific video.

---

### GET /stitcher/video/next-frame/{session_id}

**Summary:** Advances to the next frame after skips

**Description:** Advance to the next frame in the loaded video session and return metadata.

---

### GET /stitcher/video/previous-frame/{session_id}

**Summary:** Rewinds to the previous frame

**Description:** Rewind to the previous frame in the loaded video session and return metadata.

---

### GET /stitcher/video/frames/{session_id}/{frame_id}/metadata

**Summary:** Get frame navigation metadata

**Description:** Get navigation and session metadata for a frame (no detection data).

---

### GET /stitcher/video/frames/{session_id}/{frame_id}/image

**Summary:** Get raw frame image

**Description:** Stream raw frame image in PNG or JPEG format for client-side annotation.

---

### GET /stitcher/video/frames/{session_id}/cache-stats

**Summary:** Get frame cache statistics

**Description:** Get performance statistics for the rolling frame cache (hit rate, prefetch effectiveness, etc.)

---

### GET /stitcher/video/frames/{session_id}/{frame_id}/annotations

**Summary:** Get frame annotation data (detections + rendering config)

**Description:** Get supervision.Detections and rendering configuration for a frame. This is the single source of truth for annotation data.

---

### PUT /stitcher/video/frames/{session_id}/{frame_id}/annotations

**Summary:** Update frame annotation data

**Description:** Update detections and rendering configuration for a frame. This allows the web app to modify player mappings and other annotation data.

---

### GET /stitcher/video/players/{session_id}

**Summary:** Get list of players in the session

**Description:** Retrieve the list of players currently tracked in the video session.

---

### POST /stitcher/video/player/{session_id}

**Summary:** Add a player to the session

**Description:** Add a new player to the video session.

---

### PATCH /stitcher/video/player/{session_id}/{player_id}

**Summary:** Update a player

**Description:** Update an existing player

---

### DELETE /stitcher/video/player/{session_id}/{player_id}

**Summary:** Delete a player from the session

**Description:** Delete a player from the video session.

---

### DELETE /stitcher/video/stop-and-save/{session_id}

**Summary:** Stops the session and saves the processed video

**Description:** Stop the session and save the processed video.

---

