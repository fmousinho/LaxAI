
import json
import collections

def check_duplicate_tracks(tracks_path):
    print(f"Loading {tracks_path}...")
    with open(tracks_path, 'r') as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    duplicates = 0
    
    for frame in frames:
        fid = frame['frame_id']
        tids = [obj['track_id'] for obj in frame.get('track_objects', []) if obj['track_id'] >= 0]
        counts = collections.Counter(tids)
        
        has_dup = False
        for tid, count in counts.items():
            if count > 1:
                if not has_dup:
                    print(f"Frame {fid} has duplicates:")
                    has_dup = True
                print(f"  Track ID {tid} appears {count} times")
                duplicates += 1
                
    if duplicates == 0:
        print("No duplicate track IDs found in any frame.")
    else:
        print(f"Found duplicates in {duplicates} instances.")

if __name__ == "__main__":
    check_duplicate_tracks("tracks.json")
