import logging
import os
import datetime
from typing import List
import cv2

from modules.player import Player

logger = logging.getLogger(__name__)

REPORTS_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analyse", "reports")


def generate_player_report_html(run_id: str, run_output_dir: str, players: List[Player]):
    """Generates an HTML report summarizing each unique player."""
    os.makedirs(run_output_dir, exist_ok=True)
    crops_dir = os.path.join(run_output_dir, "crops")
    report_html_path = os.path.join(run_output_dir, "report.html")

    # Save all player crops to disk and store their relative paths for the report
    for player in players:
        player_crop_dir = os.path.join(crops_dir, f"player_{player.id}")
        os.makedirs(player_crop_dir, exist_ok=True)
        
        # Temporarily attach the list of saved crop paths to the player object
        player.report_crop_paths = []
        for i, crop_np in enumerate(player.crops):
            if crop_np.size == 0:
                continue
            crop_filename = f"crop_{i}.png"
            crop_abs_path = os.path.join(player_crop_dir, crop_filename)
            cv2.imwrite(crop_abs_path, crop_np)
            
            relative_path = os.path.join("crops", f"player_{player.id}", crop_filename)
            player.report_crop_paths.append(relative_path)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Player Analysis Report - Run {run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 3px #ccc; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }}
        th {{ background-color: #e9e9e9; }}
        td:nth-child(1) {{ width: 10%; font-weight: bold; text-align: center; }}
        td:nth-child(2) {{ width: 20%; }}
        .crop-gallery {{ display: flex; flex-wrap: wrap; gap: 5px; }}
        .crop-gallery img {{ max-width: 80px; max-height: 80px; border: 1px solid #eee; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Player Analysis Report - Run ID: {run_id}</h1>
    
    <table>
        <thead>
            <tr>
                <th>Player ID</th>
                <th>Associated Tracker IDs</th>
                <th>Player Crops</th>
            </tr>
        </thead>
        <tbody>
"""
    sorted_players = sorted(players, key=lambda p: p.id)

    for player in sorted_players:
        tracker_ids_str = ", ".join(map(str, sorted(player.associated_tracker_ids)))
        
        html_content += f"""
            <tr>
                <td>{player.id}</td>
                <td>{tracker_ids_str}</td>
                <td>
                    <div class="crop-gallery">
"""
        for crop_path in player.report_crop_paths:
            html_content += f'                        <img src="{crop_path}" alt="Crop for player {player.id}">\n'
        
        html_content += """
                    </div>
                </td>
            </tr>
"""

    html_content += """
        </tbody>
    </table>
</body>
</html>
"""
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Generated Player Analysis HTML report for run {run_id} at {report_html_path}")


def update_main_index_html():
    """Generates or updates the main index.html listing all runs."""
    os.makedirs(REPORTS_BASE_DIR, exist_ok=True)
    index_html_path = os.path.join(REPORTS_BASE_DIR, "index.html")

    run_ids = [item for item in os.listdir(REPORTS_BASE_DIR) if os.path.isdir(os.path.join(REPORTS_BASE_DIR, item))]
    run_ids.sort(reverse=True)

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Runs</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        ul { list-style-type: none; padding: 0; }
        li { background-color: #fff; margin: 10px 0; padding: 15px; border-radius: 5px; box-shadow: 0 2px 3px #ccc; }
        a { text-decoration: none; color: #007bff; font-weight: bold; font-size: 1.1em; }
        a:hover { text-decoration: underline; color: #0056b3; }
    </style>
</head>
<body>
    <h1>Available Analysis Runs</h1>
"""
    if not run_ids:
        html_content += "    <p>No analysis runs found.</p>\n"
    else:
        html_content += "    <ul>\n"
        for run_id in run_ids:
            html_content += f'        <li><a href="{run_id}/report.html">Run - {run_id}</a></li>\n'
        html_content += "    </ul>\n"
    
    html_content += """
</body>
</html>
"""
    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Updated main index HTML at {index_html_path}")