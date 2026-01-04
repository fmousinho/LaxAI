
import json
import collections
import sys

def verify_unique_players_per_frame(tracks_path, players_path):
    print(f"Loading tracks from {tracks_path}...")
    with open(tracks_path, 'r') as f:
        tracks_data = json.load(f)

    print(f"Loading players form {players_path}...")
    with open(players_path, 'r') as f:
        players_data = json.load(f)
    
    # Extract track_to_player mapping
    # Reconstruct track_to_player mapping from 'players' dictionary
    track_to_player = {}
    players_dict = players_data.get('players', {})
    for pid, p_data in players_dict.items():
        # p_data is a dict with 'player_id', 'team_id', 'track_ids', etc.
        player_id = int(p_data['player_id'])
        for tid in p_data.get('track_ids', []):
            track_to_player[int(tid)] = player_id
    
    print(f"Reconstructed mapping for {len(track_to_player)} tracks.")
    
    frames_data = tracks_data.get('frames', [])
    
    duplicates_found = 0
    total_frames = len(frames_data)
    
    print(f"Verifying {total_frames} frames...")
    
    for frame_data in frames_data:
        frame_id = frame_data['frame_id']
        track_objects = frame_data.get('track_objects', [])
        
        player_ids_in_frame = []
        
        for obj in track_objects:
            track_id = int(obj['track_id'])
            if track_id < 0:
                continue
                
            if track_id in track_to_player:
                player_id = track_to_player[track_id]
                player_ids_in_frame.append(player_id)
        
        # Check for duplicates
        counts = collections.Counter(player_ids_in_frame)
        for pid, count in counts.items():
            if count > 1:
                print(f"Frame {frame_id}: Player {pid} assigned to {count} tracks!")
                duplicates_found += 1
                
    if duplicates_found == 0:
        print("SUCCESS: No duplicate player assignments found in any frame.")
        sys.exit(0)
    else:
        print(f"FAILURE: Found duplicates in {duplicates_found} frames.")
        sys.exit(1)

if __name__ == "__main__":
    verify_unique_players_per_frame("tracks.json", "players_refined.json")
