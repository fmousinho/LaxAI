#!/usr/bin/env python3
"""
Demo script showing how to resume TrackStitcher from a saved graph.
"""

import networkx as nx
from supervision import Detections
import numpy as np

# Mock detections data (same as test fixture)
mock_detections = Detections(
    xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50], [60, 60, 70, 70]], dtype=np.float32),
    data={
        'frame_index': np.array([1, 2, 3, 4], dtype=np.int32),
        'tracker_id': np.array([1, 1, 2, 2], dtype=np.int32),
        'class_id': np.array([0, 0, 0, 0], dtype=np.int32)  # 4 elements to match detections
    }
)

def demo_resume_functionality():
    """Demonstrate saving and resuming stitcher state."""
    print("🚀 Demo: TrackStitcher Resume Functionality")
    print("=" * 50)

    # Import here to avoid circular imports
    from services.service_dataprep.stitcher import TrackStitcher

    # 1. Create initial stitcher and do some work
    print("\n1. Creating initial stitcher and doing some verification...")
    stitcher1 = TrackStitcher(detections=mock_detections)

    # Get first pair and merge them
    result = stitcher1.get_pair_for_verification()
    if result["status"] == "pending_verification":
        print(f"   Got pair: groups {result['group1_id']} and {result['group2_id']}")
        stitcher1.respond("same")
        print("   ✓ Merged groups")

    # Get progress
    progress = stitcher1.get_verification_progress()
    print(f"   Progress: {progress['progress_percentage']:.1f}% complete")
    print(f"   Player groups: {progress['current_player_groups']}")

    # 2. Save the current state
    print("\n2. Saving current state to checkpoint...")
    success = stitcher1.save_graph("checkpoint_demo.graphml")
    if success:
        print("   ✓ State saved successfully")
    else:
        print("   ✗ Failed to save state")
        return

    # 3. Create new stitcher instance from saved graph
    print("\n3. Resuming from saved checkpoint...")
    saved_graph = nx.read_graphml("checkpoint_demo.graphml")
    stitcher2 = TrackStitcher(detections=mock_detections, existing_graph=saved_graph)

    # Verify state was restored
    progress2 = stitcher2.get_verification_progress()
    print(f"   ✓ Resumed with {progress2['progress_percentage']:.1f}% progress")
    print(f"   ✓ Player groups: {progress2['current_player_groups']}")

    # Continue verification
    print("\n4. Continuing verification from checkpoint...")
    result = stitcher2.get_pair_for_verification()
    if result["status"] == "pending_verification":
        print(f"   Next pair: groups {result['group1_id']} and {result['group2_id']}")
        stitcher2.respond("different")
        print("   ✓ Marked as different players")

    final_progress = stitcher2.get_verification_progress()
    print(f"   Final progress: {final_progress['progress_percentage']:.1f}% complete")

    print("\n✅ Demo completed successfully!")
    print("\nThe TrackStitcher can now resume interrupted verification sessions!")

if __name__ == "__main__":
    demo_resume_functionality()