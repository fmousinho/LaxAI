# API Routing Architecture Guide

## Overview

LaxAI uses a **proxy pattern** where `service_api` (deployed as `laxai-api`) acts as the API gateway that proxies requests to backend microservices.

```
Client → service_api (laxai-api) → service_stitcher
         (Port 8080)                (Internal Cloud Run)
```

## Complete URL Structure

### Production URLs

#### Main API Gateway (service_api)
- **Service Name:** `laxai-api`
- **Base URL:** `https://laxai-api-517529966392.us-central1.run.app`
- **Health Check:** `GET /health`
- **API Docs:** `GET /docs`

#### Video/Stitching Endpoints (proxied to service_stitcher)
- **Base Path:** `/api/v1/stitch/`
- **Full URL Pattern:** `https://laxai-api-517529966392.us-central1.run.app/api/v1/stitch/{endpoint}`

### Available Endpoints

#### 1. Load Video
```http
POST /api/v1/stitch/video/load
Content-Type: application/json

{
  "tenant_id": "your-tenant-id",
  "video_path": "path/to/video.mp4"
}
```

**Response:**
```json
{
  "session_id": "uuid-session-id",
  "total_frames": 1000,
  "fps": 30.0,
  "width": 1920,
  "height": 1080
}
```

#### 2. Get Frame Metadata
```http
GET /api/v1/stitch/video/{session_id}/frames/{frame_id}/metadata
```

#### 3. Get Frame Image
```http
GET /api/v1/stitch/video/{session_id}/frames/{frame_id}/image?format=jpeg&quality=85
```

**Query Parameters:**
- `format`: `jpeg` or `png` (default: `jpeg`)
- `quality`: 1-100 (default: 85, only for JPEG)

**Response:** Binary image data

#### 4. Get Frame Annotations
```http
GET /api/v1/stitch/video/{session_id}/frames/{frame_id}/annotations
```

**Response:**
```json
{
  "frame_id": 0,
  "detections": {
    "xyxy": [[x1, y1, x2, y2], ...],
    "confidence": [0.95, ...],
    "class_id": [0, ...],
    "tracker_id": [1, ...]
  },
  "rendering_config": {
    "player_styles": {},
    "tracker_styles": {}
  }
}
```

#### 5. Update Frame Annotations
```http
PUT /api/v1/stitch/video/{session_id}/frames/{frame_id}/annotations
Content-Type: application/json

{
  "detections": {
    "xyxy": [[x1, y1, x2, y2], ...],
    "tracker_id": [1, ...]
  },
  "rendering_config": {
    "player_styles": {
      "12": {
        "base_color": [255, 0, 0],
        "thickness": 2
      }
    }
  }
}
```

#### 6. Stop and Save Session
```http
POST /api/v1/stitch/video/{session_id}/stop
```

## Internal Routing Flow

### Request Path Translation

```
Client Request:
POST /api/v1/stitch/video/load

↓ service_api (laxai-api)
  - Main app receives: /api/v1/stitch/video/load
  - Router prefix: /api/v1
  - Stitch router prefix: /stitch
  - Catch-all matches: video/load
  
↓ Proxy forwards to service_stitcher
  - Strips /stitch prefix
  - Forwards: /api/v1/video/load

↓ service_stitcher receives:
  - Main app: /api/v1/video/load
  - Router prefix: /api/v1
  - Video router prefix: /video
  - Endpoint: /load
  - ✅ Match!
```

## Common Errors

### ❌ 404 Not Found - Wrong Path

**Log:**
```
POST /video/load HTTP/1.1" 404
```

**Problem:** Client is calling `/video/load` instead of `/api/v1/stitch/video/load`

**Solution:** Update client to use full path with `/api/v1/stitch/` prefix

**Correct Client Code:**
```javascript
// ❌ WRONG
const response = await fetch('https://laxai-api.../video/load', {...})

// ✅ CORRECT
const response = await fetch('https://laxai-api.../api/v1/stitch/video/load', {...})
```

### ❌ 401 Unauthorized - Missing Authentication

**Problem:** Request missing GCP service account authentication

**Solution:** Ensure Node.js backend is using GCP service account credentials

```javascript
const { GoogleAuth } = require('google-auth-library');

const auth = new GoogleAuth({
  scopes: ['https://www.googleapis.com/auth/cloud-platform']
});
const client = await auth.getClient();
const token = await client.getAccessToken();

// Add to request headers
headers['Authorization'] = `Bearer ${token.token}`;
```

### ❌ 503 Service Unavailable - service_stitcher Down

**Problem:** Backend service not running or not accessible

**Solution:** 
1. Check service_stitcher deployment status
2. Verify service account has permission to invoke service_stitcher
3. Check Cloud Run logs for service_stitcher

## Testing Endpoints

### Using curl with gcloud auth

```bash
# Get auth token
TOKEN=$(gcloud auth print-identity-token)

# Test load video
curl -X POST \
  "https://laxai-api-517529966392.us-central1.run.app/api/v1/stitch/video/load" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test-tenant",
    "video_path": "gs://bucket/video.mp4"
  }'
```

### Using Node.js with google-auth-library

```javascript
const { GoogleAuth } = require('google-auth-library');
const axios = require('axios');

async function callStitchAPI() {
  // Initialize auth
  const auth = new GoogleAuth({
    scopes: ['https://www.googleapis.com/auth/cloud-platform']
  });
  const client = await auth.getClient();
  const token = await client.getAccessToken();
  
  // Make request
  const response = await axios.post(
    'https://laxai-api-517529966392.us-central1.run.app/api/v1/stitch/video/load',
    {
      tenant_id: 'test-tenant',
      video_path: 'gs://bucket/video.mp4'
    },
    {
      headers: {
        'Authorization': `Bearer ${token.token}`,
        'Content-Type': 'application/json'
      }
    }
  );
  
  return response.data;
}
```

## Environment Variables

### service_api (laxai-api)
- `SERVICE_STITCH_URL`: Override service_stitcher URL (optional, defaults to Cloud Run URL)
- `PORT`: HTTP port (default: 8080)
- `CORS_ORIGINS`: Allowed CORS origins (default: "*")

### service_stitcher
- `PORT`: HTTP port (default: 8080)
- `CORS_ORIGINS`: Allowed CORS origins (default: "*")

## Deployment

### Deploy service_api
```bash
gcloud builds submit . \
  --config=deploy/cloudbuild/cloudbuild-api.yaml \
  --region=us-central1
```

### Deploy service_stitcher
```bash
gcloud builds submit . \
  --config=deploy/cloudbuild/cloudbuild-service-stitcher.yaml \
  --region=us-central1
```

## Debugging Checklist

When you get a 404:
1. ✅ Check the full URL path includes `/api/v1/stitch/`
2. ✅ Verify authorization header is present
3. ✅ Check service_api logs for routing errors
4. ✅ Check service_stitcher logs to see if request arrived
5. ✅ Verify both services are deployed and running

When you get a 401:
1. ✅ Verify GCP service account authentication is working
2. ✅ Check service account has Cloud Run Invoker role
3. ✅ Ensure token is included in Authorization header

When you get a 503:
1. ✅ Check service_stitcher is running
2. ✅ Verify service_stitcher is accessible from service_api
3. ✅ Check Cloud Run logs for both services
4. ✅ Verify service account permissions
