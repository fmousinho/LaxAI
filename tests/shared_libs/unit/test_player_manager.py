import json
import pytest
from pathlib import Path
import tempfile

from shared_libs.common.player_manager import Player, PlayerManager


class TestPlayerManager:
    """Unit tests for PlayerManager class."""

    def test_serialize_and_load_synthetic_players(self):
        """Test serialization and loading of synthetic player data."""
        # Create a PlayerManager for testing
        video_id = "test_video_123"
        manager = PlayerManager(video_id)

        # Create 10 synthetic players with names and track associations
        player_names = [
            "Alice", "Bob", "Charlie", "Diana", "Eve",
            "Frank", "Grace", "Henry", "Ivy", "Jack"
        ]

        created_players = []
        for i, name in enumerate(player_names):
            player = manager.create_player(name=name)
            created_players.append(player)

            # Add some track associations (each player gets 2-3 tracks)
            for track_offset in range(2 + (i % 2)):  # 2 or 3 tracks per player
                track_id = i * 10 + track_offset + 1
                manager.add_track_to_player(player.id, track_id)

        # Verify we have 10 players
        assert len(manager.get_all_players()) == 10

        # Serialize to JSON
        json_data = manager.serialize_to_save()
        assert json_data, "Serialization should produce non-empty JSON"

        # Verify JSON structure
        data = json.loads(json_data)
        assert data["video_id"] == video_id
        assert len(data["players"]) == 10
        assert "track_to_player" in data

        # Create a new PlayerManager and load from JSON
        new_manager = PlayerManager(video_id)
        new_manager.load_players_from_json(json_data)

        # Verify all players were loaded
        loaded_players = new_manager.get_all_players()
        assert len(loaded_players) == 10

        # Verify each player has correct data
        loaded_players.sort(key=lambda p: p.id)
        created_players.sort(key=lambda p: p.id)

        for original, loaded in zip(created_players, loaded_players):
            assert loaded.id == original.id
            assert loaded.name == original.name
            assert set(loaded.track_ids) == set(original.track_ids)

            # Verify track-to-player mappings
            for track_id in loaded.track_ids:
                mapped_player = new_manager.get_player_by_track_id(track_id)
                assert mapped_player is not None
                assert mapped_player.id == loaded.id

        # Verify track_to_player mapping size
        expected_tracks = sum(len(p.track_ids) for p in created_players)
        assert len(new_manager.track_to_player) == expected_tracks

    def test_serialization_with_empty_manager(self):
        """Test serialization of an empty PlayerManager."""
        video_id = "empty_video"
        manager = PlayerManager(video_id)

        # No players created
        assert len(manager.get_all_players()) == 0

        # Serialize
        json_data = manager.serialize_to_save()
        data = json.loads(json_data)

        assert data["video_id"] == video_id
        assert len(data["players"]) == 0
        assert data["track_to_player"] == {}

        # Load into new manager
        new_manager = PlayerManager(video_id)
        new_manager.load_players_from_json(json_data)

        assert len(new_manager.get_all_players()) == 0

    def test_serialization_with_partial_data(self):
        """Test serialization with players that have some missing optional fields."""
        video_id = "partial_video"
        manager = PlayerManager(video_id)

        # Create players with varying data
        player1 = manager.create_player(name="Named Player")
        player2 = manager.create_player()  # No name
        manager.add_track_to_player(player1.id, 100)
        # player2 has no tracks

        # Serialize and load
        json_data = manager.serialize_to_save()
        new_manager = PlayerManager(video_id)
        new_manager.load_players_from_json(json_data)

        loaded_players = new_manager.get_all_players()
        assert len(loaded_players) == 2

        # Find players by ID
        loaded_player1 = new_manager.get_player_by_id(player1.id)
        loaded_player2 = new_manager.get_player_by_id(player2.id)

        assert loaded_player1 is not None
        assert loaded_player2 is not None

        assert loaded_player1.name == "Named Player"
        assert loaded_player1.track_ids == [100]
        assert loaded_player2.name is None
        assert loaded_player2.track_ids == []