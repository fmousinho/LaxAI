import logging
import os
import cv2

logger = logging.getLogger(__name__)

def generate_track_report_html(run_id: str, run_output_dir: str, track_rows: list):
    """Generates an HTML report with a row for each track, showing crops, masked crops, team, and player ID."""
    os.makedirs(run_output_dir, exist_ok=True)
    crops_dir = os.path.join(run_output_dir, "crops")
    masked_dir = os.path.join(run_output_dir, "masked_crops")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)
    report_html_path = os.path.join(run_output_dir, "report.html")

    # Save crops and masked crops, collect their relative paths
    for row in track_rows:
        tid = row["track_id"]
        crop = row["original_crop"]
        masked_crop = row["masked_crop"]
        crop_path = None
        masked_path = None
        if crop is not None and hasattr(crop, 'size') and crop.size > 0:
            crop_filename = f"crop_{tid}.png"
            crop_abs_path = os.path.join(crops_dir, crop_filename)
            cv2.imwrite(crop_abs_path, crop)
            crop_path = os.path.join("crops", crop_filename)
        if masked_crop is not None and hasattr(masked_crop, 'size') and masked_crop.size > 0:
            masked_filename = f"masked_{tid}.png"
            masked_abs_path = os.path.join(masked_dir, masked_filename)
            cv2.imwrite(masked_abs_path, masked_crop)
            masked_path = os.path.join("masked_crops", masked_filename)
        row["crop_path"] = crop_path
        row["masked_path"] = masked_path

    html_content = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Track Analysis Report - Run {run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 3px #ccc; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }}
        th {{ background-color: #e9e9e9; }}
        .crop-img {{ max-width: 80px; max-height: 80px; border: 1px solid #eee; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Track Analysis Report - Run ID: {run_id}</h1>
    <table>
        <thead>
            <tr>
                <th>Track ID</th>
                <th>Original Crop</th>
                <th>Masked Crop</th>
                <th>Team</th>
                <th>Player ID</th>
            </tr>
        </thead>
        <tbody>
"""
    for row in track_rows:
        html_content += f"<tr>"
        html_content += f"<td>{row['track_id']}</td>"
        if row["crop_path"]:
            html_content += f'<td><img src="{row["crop_path"]}" class="crop-img"></td>'
        else:
            html_content += f'<td></td>'
        if row["masked_path"]:
            html_content += f'<td><img src="{row["masked_path"]}" class="crop-img"></td>'
        else:
            html_content += f'<td></td>'
        html_content += f"<td>{row['team']}</td>"
        html_content += f"<td>{row['player_id']}</td>"
        html_content += f"</tr>"
    html_content += """
        </tbody>
    </table>
</body>
</html>
"""
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Generated Track Analysis HTML report for run {run_id}:")
    logger.info (f"link: {report_html_path}")


logger = logging.getLogger(__name__)

REPORTS_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analyse", "reports")



def generate_player_report_html(run_id: str, run_output_dir: str, player_rows: list):
    """Generates an HTML report summarizing each unique player (dict-based, not Player objects)."""
    os.makedirs(run_output_dir, exist_ok=True)
    crops_dir = os.path.join(run_output_dir, "crops")
    report_html_path = os.path.join(run_output_dir, "report.html")

    # Save all player crops to disk and store their relative paths for the report
    for row in player_rows:
        player_id = row["player_id"]
        player_crop_dir = os.path.join(crops_dir, f"player_{player_id}")
        os.makedirs(player_crop_dir, exist_ok=True)
        crop_paths = []
        for i, crop_np in enumerate(row["crops"]):
            if crop_np is None or not hasattr(crop_np, 'size') or crop_np.size == 0:
                continue
            crop_filename = f"crop_{i}.png"
            crop_abs_path = os.path.join(player_crop_dir, crop_filename)
            cv2.imwrite(crop_abs_path, crop_np)
            relative_path = os.path.join("crops", f"player_{player_id}", crop_filename)
            crop_paths.append(relative_path)
        row["report_crop_paths"] = crop_paths

    html_content = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
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
    sorted_rows = sorted(player_rows, key=lambda r: r["player_id"])
    for row in sorted_rows:
        tracker_ids_str = ", ".join(map(str, sorted(row["tracker_ids"])))
        html_content += f"""
            <tr>
                <td>{row['player_id']}</td>
                <td>{tracker_ids_str}</td>
                <td>
                    <div class=\"crop-gallery\">
"""
        for crop_path in row["report_crop_paths"]:
            html_content += f'                        <img src="{crop_path}" alt="Crop for player {row["player_id"]}">\n'
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