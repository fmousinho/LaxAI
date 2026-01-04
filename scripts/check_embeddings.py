import json
import torch

# Load tracks and embeddings
with open('tracks.json', 'r') as f:
    tracks_data = json.load(f)

embeddings_data = torch.load('emb.pt')

# Count tracks
frames = tracks_data.get('frames', [])
all_track_ids = set()
for frame in frames:
    for obj in frame.get('track_objects', []):
        tid = obj['track_id']
        if tid >= 0:
            all_track_ids.add(tid)

print(f"Total unique tracks: {len(all_track_ids)}")
print(f"Tracks with embeddings: {len(embeddings_data)}")
print(f"Coverage: {100*len(embeddings_data)/len(all_track_ids):.1f}%")

# Check embedding quality
tracks_with_all = 0
tracks_with_mean = 0
for tid, emb_data in embeddings_data.items():
    if isinstance(emb_data, dict):
        if 'all' in emb_data and emb_data['all'] is not None:
            tracks_with_all += 1
        if 'mean' in emb_data and emb_data['mean'] is not None:
            tracks_with_mean += 1

print(f"\nTracks with 'all' embeddings: {tracks_with_all}")
print(f"Tracks with 'mean' embeddings: {tracks_with_mean}")
