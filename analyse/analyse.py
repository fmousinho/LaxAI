import logging
import os
import datetime
import json
from typing import List, Dict, Any
import cv2 # For saving images

import torch
import supervision as sv
from tqdm import tqdm

from ..modules.detection import DetectionModel
from ..tools.store_driver import Store

logger = logging.getLogger(__name__)

# Define the base directory for reports, relative to this script file.
# This will create 'reports' inside the 'analyse' module directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_BASE_DIR = os.path.join(SCRIPT_DIR, "reports")


def _generate_run_html_report(run_id: str, run_output_dir: str, crops_data: List[Dict[str, Any]]):
    """Generates the HTML report for a single run."""
    os.makedirs(run_output_dir, exist_ok=True)
    report_html_path = os.path.join(run_output_dir, "report.html")

    # Group crops by frame number
    frames_data: Dict[int, List[Dict[str, Any]]] = {}
    for crop_info in crops_data:
        frame_num = crop_info["frame_number"]
        if frame_num not in frames_data:
            frames_data[frame_num] = []
        frames_data[frame_num].append(crop_info)

    sorted_frame_numbers = sorted(frames_data.keys())

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Report - Run {run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        h1, h3 {{ color: #333; }}
        .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; margin-bottom: 10px; border-radius: 5px; }}
        .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 16px; }}
        .tab button:hover {{ background-color: #ddd; }}
        .tab button.active {{ background-color: #ccc; font-weight: bold; }}
        .tabcontent {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; background-color: #fff; border-radius: 0 0 5px 5px; }}
        .tabcontent.active {{ display: block; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 3px #ccc; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #e9e9e9; }}
        img.crop {{ max-width: 200px; max-height: 200px; border: 1px solid #eee; border-radius: 4px; display: block; margin: auto; }}
        td:first-child {{ width: 220px; text-align: center; }}
    </style>
</head>
<body>
    <h1>Analysis Report - Run ID: {run_id}</h1>

    <div class="tab">
"""
    for i, frame_num in enumerate(sorted_frame_numbers):
        active_class = "active" if i == 0 else ""
        html_content += f'        <button class="tablinks {active_class}" onclick="openFrame(event, \'Frame{frame_num}\')">Frame {frame_num}</button>\n'
    html_content += "    </div>\n\n"

    for i, frame_num in enumerate(sorted_frame_numbers):
        active_class = "active" if i == 0 else ""
        html_content += f'    <div id="Frame{frame_num}" class="tabcontent {active_class}">\n'
        html_content += f"        <h3>Detections for Frame {frame_num}</h3>\n"
        if not frames_data[frame_num]:
            html_content += "<p>No detections for this frame.</p>"
        else:
            html_content += """
        <table>
            <thead>
                <tr>
                    <th>Cropped Image</th>
                    <th>Confidence</th>
                    <th>Class ID</th>
                    <th>Track ID</th>
                    <th>Bounding Box (xyxy)</th>
                </tr>
            </thead>
            <tbody>
"""
            for crop_info in frames_data[frame_num]:
                confidence_str = f"{crop_info['confidence']:.2f}" if crop_info['confidence'] is not None else "N/A"
                class_id_str = str(crop_info['class_id']) if crop_info['class_id'] is not None else "N/A"
                track_id_str = str(crop_info['track_id']) if crop_info['track_id'] is not None else "N/A"
                bbox_str = str(crop_info['bbox_xyxy'])
                img_path = crop_info['crop_image_path'] # Relative to report.html

                html_content += f"""
                <tr>
                    <td><img src="{img_path}" alt="Crop from frame {frame_num}" class="crop"></td>
                    <td>{confidence_str}</td>
                    <td>{class_id_str}</td>
                    <td>{track_id_str}</td>
                    <td>{bbox_str}</td>
                </tr>
"""
            html_content += """
            </tbody>
        </table>
"""
        html_content += "    </div>\n" # End tabcontent

    html_content += """
    <script>
        function openFrame(evt, frameName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
                tabcontent[i].classList.remove("active");
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].classList.remove("active");
            }
            document.getElementById(frameName).style.display = "block";
            document.getElementById(frameName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }
        // Automatically open the first tab if it exists
        if (document.getElementsByClassName("tablinks").length > 0) {
            document.getElementsByClassName("tablinks")[0].click();
        }
    </script>
</body>
</html>
"""
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Generated HTML report for run {run_id} at {report_html_path}")


def _update_main_index_html(reports_base_dir: str):
    """Generates or updates the main index.html listing all runs."""
    os.makedirs(reports_base_dir, exist_ok=True)
    index_html_path = os.path.join(reports_base_dir, "index.html")

    run_ids = []
    if os.path.exists(reports_base_dir):
        for item in os.listdir(reports_base_dir):
            item_path = os.path.join(reports_base_dir, item)
            # Ensure item is a directory and contains a report.html (or run_data.json)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "report.html")):
                run_ids.append(item)
    
    run_ids.sort(reverse=True) # Sort by timestamp, newest first

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
        p { font-size: 1.1em; }
    </style>
</head>
<body>
    <h1>Available Analysis Runs</h1>
"""
    if not run_ids:
        html_content += "    <p>No analysis runs found. Process a video to generate a report.</p>\n"
    else:
        html_content += "    <ul>\n"
        for run_id in run_ids:
            # Link is relative to index.html, which is in reports_base_dir
            html_content += f'        <li><a href="{run_id}/report.html">Run - {run_id}</a></li>\n'
        html_content += "    </ul>\n"
    
    html_content += """
</body>
</html>
"""
    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Updated main index HTML at {index_html_path}")


def analyse_video(
        store: Store,
        input_video: str,
        start_frame: int,
        end_frame: int,
        device: torch.device = torch.device("cpu"),
    ) -> str:
    """
    Analyses a video, extracts detections, saves crop images and data,
    and generates an HTML report.
    Args:
        store: The Store instance for file management (used for model loading).
        input_video: Path to the input video file.
        start_frame: The starting frame number for processing.
        end_frame: The ending frame number for processing (exclusive).
        device: The torch device to use for processing (e.g., "cpu", "cuda").
    Returns:
        str: Path to the generated HTML report for the run.
    """
    # Generate a unique run ID (timestamp)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_output_dir = os.path.join(REPORTS_BASE_DIR, run_id)
    crops_img_dir = os.path.join(run_output_dir, "crops")

    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(crops_img_dir, exist_ok=True)
    logger.info(f"Starting analysis run {run_id}. Output will be in {run_output_dir}")

    video_info = sv.VideoInfo.from_video_path(video_path=input_video)
    frames_generator = sv.get_video_frames_generator(source_path=input_video, start=start_frame, end=end_frame)
    model = DetectionModel(store=store, device=device)

    tracker = sv.ByteTrack(
        track_activation_threshold = 0.25,
        lost_track_buffer = 30,
        minimum_matching_threshold=0.8,
        frame_rate = video_info.fps,
        minimum_consecutive_frames=15
    )
    
    actual_total_frames_to_process = end_frame - start_frame
    processed_crops_data: List[Dict[str, Any]] = []
    current_frame_number = start_frame

    for frame_idx, frame in enumerate(tqdm(frames_generator, desc="Processing frames", total=actual_total_frames_to_process)):
        detections = model.generate_detections(frame)
        detections = tracker.update_with_detections(detections)

        # If 'detections' is an iterable of tuples (e.g., from a custom model output):
        for det_idx, detection_item in enumerate(detections):
            # Assuming detection_item is a tuple, e.g., (bbox_xyxy_list, confidence_float, class_id_int)
            
            # Extract bounding box coordinates and convert them to integers
            # Assuming the first element of the tuple is the bbox coordinates
            x1, y1, x2, y2 = map(int, detection_item[0])

            # Crop the image from the frame using NumPy array slicing (BGR)
            crop_img_to_save = frame[y1:y2, x1:x2]

            if crop_img_to_save.size == 0:
                logger.warning(f"Empty crop generated for frame {current_frame_number}, bbox {(x1, y1, x2, y2)}. Skipping.")
                continue
            
            crop_filename = f"frame_{current_frame_number}_det_{det_idx}.png"
            crop_image_abs_path = os.path.join(crops_img_dir, crop_filename)
            
            try:
                cv2.imwrite(crop_image_abs_path, crop_img_to_save)
            except Exception as e:
                logger.error(f"Failed to save crop image {crop_image_abs_path}: {e}")
                continue # Skip this crop if saving failed

            # Store relative path for HTML (relative to run_output_dir) and JSON
            crop_image_relative_path = os.path.join("crops", crop_filename)

            # Assuming the second element is confidence and third is class_id
            confidence_val = detection_item[2] if len(detection_item) > 2 else None
            class_id_val = detection_item[3] if len(detection_item) > 3 else None
            track_id = detection_item[4]

            processed_crops_data.append({
                "crop_image_path": crop_image_relative_path,
                "confidence": float(confidence_val) if confidence_val is not None else None,
                "class_id": int(class_id_val) if class_id_val is not None else None,
                "track_id": int(track_id) if track_id is not None else None,
                "bbox_xyxy": (x1, y1, x2, y2),
                "frame_number": current_frame_number # Absolute frame number
            })
        current_frame_number += 1

    # Save processed_crops_data to JSON
    run_data_json_path = os.path.join(run_output_dir, "run_data.json")
    with open(run_data_json_path, "w", encoding="utf-8") as f:
        json.dump(processed_crops_data, f, indent=4)
    logger.info(f"Saved run data to {run_data_json_path}")

    # Generate HTML report for this run
    _generate_run_html_report(run_id, run_output_dir, processed_crops_data)

    # Update the main index.html
    _update_main_index_html(REPORTS_BASE_DIR)

    report_path = os.path.join(run_output_dir, 'report.html')
    logger.info(f"Analysis run {run_id} completed. Report at {report_path}")
    return report_path
