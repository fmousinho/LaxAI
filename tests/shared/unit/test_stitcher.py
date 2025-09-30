import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch
from supervision import Detections

from services.service_dataprep.src.stitcher import TrackStitcher, EdgeType


class TestTrackStitcher:
    """Unit tests for TrackStitcher class."""

    @pytest.fixture
    def mock_detections(self):
        """Create a mock Detections object for testing."""
        # Create mock detections with 4 tracks across different frames
        mock_dets = Mock(spec=Detections)
        mock_dets.data = {
            'frame_index': np.array([1, 2, 1, 2, 3, 4, 3, 4], dtype=np.int32),
            'tracker_id': np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
        }
        # Set tracker_id as a direct attribute for the stitcher code
        mock_dets.tracker_id = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
        return mock_dets

    @pytest.fixture
    def stitcher(self, mock_detections):
        """Create a TrackStitcher instance for testing."""
        return TrackStitcher(detections=mock_detections)

    def test_initialization(self, stitcher, mock_detections):
        """Test that TrackStitcher initializes correctly."""
        assert stitcher.detections == mock_detections
        assert isinstance(stitcher._track_to_frames, dict)
        assert isinstance(stitcher.player_groups, dict)
        assert isinstance(stitcher.track_graph, type(stitcher.track_graph))  # NetworkX graph

        # Check that all tracks are in player_groups as individual groups
        expected_tracks = {1, 2, 3, 4}
        assert set(stitcher.player_groups.keys()) == expected_tracks
        for track_id in expected_tracks:
            assert stitcher.player_groups[track_id] == {track_id}

    def test_invert_supervision_data(self, mock_detections):
        """Test the _invert_supervision_data method."""
        stitcher = TrackStitcher.__new__(TrackStitcher)  # Create without __init__
        stitcher.detections = mock_detections

        result = stitcher._invert_supervision_data(mock_detections)

        assert isinstance(result, dict)
        assert 1 in result  # track 1
        assert 2 in result  # track 2
        assert 3 in result  # track 3
        assert 4 in result  # track 4

        # Check frame arrays are numpy arrays and sorted
        assert isinstance(result[1], np.ndarray)
        assert np.array_equal(result[1], np.array([1, 2]))  # track 1 in frames 1,2
        assert np.array_equal(result[2], np.array([1, 2]))  # track 2 in frames 1,2
        assert np.array_equal(result[3], np.array([3, 4]))  # track 3 in frames 3,4
        assert np.array_equal(result[4], np.array([3, 4]))  # track 4 in frames 3,4

    def test_tracks_overlap_in_time(self, stitcher):
        """Test temporal overlap detection between tracks."""
        # Tracks 1 and 2 overlap (both in frames 1,2)
        assert stitcher._tracks_overlap_in_time(1, 2) == True

        # Tracks 1 and 3 don't overlap (1: [1,2], 3: [3,4])
        assert stitcher._tracks_overlap_in_time(1, 3) == False

        # Tracks 3 and 4 overlap (both in frames 3,4)
        assert stitcher._tracks_overlap_in_time(3, 4) == True

    def test_groups_overlap_in_time(self, stitcher):
        """Test temporal overlap detection between groups."""
        # Initially each track is its own group
        # Groups 1 and 2 overlap (tracks 1,2 both in frames 1,2)
        assert stitcher._groups_overlap_in_time(1, 2) == True

        # Groups 1 and 3 don't overlap
        assert stitcher._groups_overlap_in_time(1, 3) == False

    def test_get_pair_for_verification_initial(self, stitcher):
        """Test getting first pair for verification."""
        result = stitcher.get_pair_for_verification()

        assert result["status"] == "pending_verification"
        assert "group1_id" in result
        assert "group2_id" in result
        assert result["mode"] == "normal"

        # Should be the first pair that doesn't have temporal conflicts
        # Groups 1 and 2 have temporal conflict, so should skip to 1 and 3
        assert result["group1_id"] == 1
        assert result["group2_id"] == 3

    def test_respond_same_player(self, stitcher):
        """Test responding with 'same' merges groups."""
        # Get a pair first - should be groups 1 and 3 (since 1&2 and 3&4 have temporal conflicts)
        result = stitcher.get_pair_for_verification()
        assert result["status"] == "pending_verification"
        group1_id = result["group1_id"]
        group2_id = result["group2_id"]

        # Should be groups that don't have temporal conflicts
        assert group1_id == 1
        assert group2_id == 3

        # Respond that they are the same player
        stitcher.respond("same")

        # Check that groups were merged
        # The smaller ID becomes the representative
        merged_group_id = min(group1_id, group2_id)
        assert merged_group_id in stitcher.player_groups
        assert group1_id not in stitcher.player_groups or group2_id not in stitcher.player_groups

        # Check that all tracks from both original groups are now in the merged group
        merged_tracks = stitcher.player_groups[merged_group_id]
        expected_tracks = {group1_id, group2_id}  # The original track IDs
        assert expected_tracks.issubset(merged_tracks)

    def test_respond_different_player(self, stitcher):
        """Test responding with 'different' records conflict."""
        # Get a pair first
        result = stitcher.get_pair_for_verification()
        group1_id = result["group1_id"]
        group2_id = result["group2_id"]

        # Respond that they are different players
        stitcher.respond("different")

        # Check graph has the relationship
        group1_tracks = stitcher.player_groups[group1_id]
        group2_tracks = stitcher.player_groups[group2_id]

        for t1 in group1_tracks:
            for t2 in group2_tracks:
                assert stitcher.track_graph.has_edge(t1, t2)
                assert stitcher.track_graph[t1][t2]['relationship'] == EdgeType.DIFFERENT_PLAYER

    def test_respond_skip(self, stitcher):
        """Test responding with 'skip' defers decision."""
        # Get a pair first
        result = stitcher.get_pair_for_verification()
        group1_id = result["group1_id"]
        group2_id = result["group2_id"]

        # Respond with skip
        stitcher.respond("skip")

        # Check graph has the relationship
        group1_tracks = stitcher.player_groups[group1_id]
        group2_tracks = stitcher.player_groups[group2_id]

        for t1 in group1_tracks:
            for t2 in group2_tracks:
                assert stitcher.track_graph.has_edge(t1, t2)
                assert stitcher.track_graph[t1][t2]['relationship'] == EdgeType.SKIPPED

        # Check it's in skipped pairs
        skipped = stitcher.get_skipped_pairs()
        assert (group1_id, group2_id) in skipped or (group2_id, group1_id) in skipped

    def test_respond_invalid_decision(self, stitcher):
        """Test that invalid decisions raise ValueError."""
        # Get a pair first
        stitcher.get_pair_for_verification()

        with pytest.raises(ValueError, match="Invalid decision 'invalid'"):
            stitcher.respond("invalid")

    def test_respond_without_pair(self, stitcher):
        """Test that responding without a proposed pair raises error."""
        with pytest.raises(RuntimeError, match="respond\\(\\) called before a pair was proposed"):
            stitcher.respond("same")

    def test_get_verification_progress(self, stitcher):
        """Test getting verification progress statistics."""
        progress = stitcher.get_verification_progress()

        assert "total_tracks" in progress
        assert "total_possible_pairs" in progress
        assert "progress_percentage" in progress
        assert "skipped_pairs" in progress
        assert "remaining_pairs" in progress

        assert progress["total_tracks"] == 4  # 4 tracks
        assert progress["total_possible_pairs"] == 6  # C(4,2) = 6
        # Progress is not 0% because temporal conflicts are already recorded
        assert progress["progress_percentage"] > 0.0

    def test_start_second_pass_no_skipped(self, stitcher):
        """Test starting second pass when no pairs were skipped."""
        # Verify some pairs first
        result = stitcher.get_pair_for_verification()
        stitcher.respond("same")

        result = stitcher.get_pair_for_verification()
        stitcher.respond("different")

        # Should complete without skipped pairs
        assert stitcher.start_second_pass() == False

    def test_start_second_pass_with_skipped(self, stitcher):
        """Test starting second pass when pairs were skipped."""
        # Skip a pair
        result = stitcher.get_pair_for_verification()
        stitcher.respond("skip")

        # Should start second pass
        assert stitcher.start_second_pass() == True
        assert stitcher._verification_mode == "second_pass"

    @patch('matplotlib.pyplot')
    @patch('PIL.Image')
    def test_visualize_graph(self, mock_image, mock_plt, stitcher):
        """Test graph visualization returns PIL Image."""
        # Mock the PIL Image
        mock_img = Mock()
        mock_image.open.return_value = mock_img

        # Mock matplotlib
        mock_plt.figure.return_value = Mock()
        mock_plt.savefig.return_value = None
        mock_plt.close.return_value = None

        result = stitcher.visualize_graph()

        assert result == mock_img
        mock_plt.figure.assert_called_once()
        mock_plt.savefig.assert_called_once()

    @patch('matplotlib.pyplot')
    def test_visualize_graph_import_error(self, mock_plt, stitcher):
        """Test visualization handles import errors gracefully."""
        mock_plt.figure.side_effect = ImportError("No module named 'matplotlib'")

        result = stitcher.visualize_graph()
        assert result is None

    def test_get_connected_components(self, stitcher):
        """Test getting connected components based on SAME_PLAYER relationships."""
        # Initially all tracks are separate components
        components = stitcher._get_connected_components()
        assert len(components) == 4  # 4 separate tracks

        # Merge some tracks
        result = stitcher.get_pair_for_verification()
        stitcher.respond("same")  # This merges groups

        components = stitcher._get_connected_components()
        # Should now have fewer components (some merged)
        assert len(components) < 4

    def test_get_skipped_pairs(self, stitcher):
        """Test getting list of skipped pairs."""
        # Initially no skipped pairs
        assert stitcher.get_skipped_pairs() == []

        # Skip a pair
        result = stitcher.get_pair_for_verification()
        group1, group2 = result["group1_id"], result["group2_id"]
        stitcher.respond("skip")

        skipped = stitcher.get_skipped_pairs()
        assert len(skipped) == 1
        assert (group1, group2) in skipped or (group2, group1) in skipped

    def test_get_group_context(self, stitcher):
        """Test getting context information about a group."""
        # Initially each group has no relationships, so context is minimal
        context = stitcher._get_group_context(1)
        assert context == "No additional context"

        # After merging, context should show merged tracks
        result = stitcher.get_pair_for_verification()
        group1, group2 = result["group1_id"], result["group2_id"]
        stitcher.respond("same")

        # Get context for the merged group
        merged_group = min(group1, group2)
        context = stitcher._get_group_context(merged_group)
        assert "original tracks" in context

    def test_get_graph(self, stitcher):
        """Test getting access to the underlying graph."""
        graph = stitcher.get_graph()
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 4  # 4 tracks
        assert graph.has_node(1)
        assert graph.has_node(2)

    def test_save_graph(self, stitcher, tmp_path):
        """Test saving graph to different formats."""
        # Test GraphML format
        graphml_path = tmp_path / "test_graph.graphml"
        result = stitcher.save_graph(str(graphml_path), "graphml")
        assert result is True
        assert graphml_path.exists()

        # Test JSON format
        json_path = tmp_path / "test_graph.json"
        result = stitcher.save_graph(str(json_path), "json")
        assert result is True
        assert json_path.exists()

        # Test invalid format
        result = stitcher.save_graph(str(tmp_path / "test.invalid"), "invalid")
        assert result is False

    def test_export_graph_data(self, stitcher):
        """Test exporting graph data as dictionary."""
        data = stitcher.export_graph_data()

        assert "nodes" in data
        assert "edges" in data
        assert "player_groups" in data
        assert "player_groups_dict" in data
        assert "metadata" in data

        assert len(data["nodes"]) == 4  # 4 tracks
        assert data["metadata"]["total_tracks"] == 4
        assert data["metadata"]["player_count"] == 4  # Initially each track is its own player

        # Check that edges include temporal conflicts
        edge_types = [edge["relationship"] for edge in data["edges"]]
        assert "temporal_conflict" in edge_types

    def test_resume_from_existing_graph(self, mock_detections, tmp_path):
        """Test resuming verification from a saved graph."""
        # Create initial stitcher and do some work
        stitcher1 = TrackStitcher(detections=mock_detections)
        
        # Get a pair and respond
        result = stitcher1.get_pair_for_verification()
        assert result["status"] == "pending_verification"
        group1_id, group2_id = result["group1_id"], result["group2_id"]
        stitcher1.respond("same")  # Merge the groups
        
        # Save the graph
        graph_path = tmp_path / "test_resume.graphml"
        success = stitcher1.save_graph(str(graph_path))
        assert success
        
        # Create new stitcher from saved graph
        import networkx as nx
        saved_graph = nx.read_graphml(str(graph_path))
        stitcher2 = TrackStitcher(detections=mock_detections, existing_graph=saved_graph)
        
        # Verify state was reconstructed correctly
        assert stitcher2.track_graph.number_of_nodes() == 4
        assert len(stitcher2.player_groups) < 4  # Should have fewer groups due to merging
        
        # Check that the merged groups are preserved
        merged_group_found = False
        for group_tracks in stitcher2.player_groups.values():
            if len(group_tracks) > 1:
                merged_group_found = True
                break
        assert merged_group_found, "Merged groups should be preserved when resuming"
        
        # Verify temporal conflicts are still present
        temporal_count = 0
        for _, _, data in stitcher2.track_graph.edges(data=True):
            if data.get('relationship') == EdgeType.TEMPORAL_CONFLICT:
                temporal_count += 1
        assert temporal_count > 0, "Temporal conflicts should be preserved when resuming"