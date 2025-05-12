import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D scatter plot
import logging
import os # Added os import
from typing import Optional, Dict
from .videotools import VideoToools # Import VideoToools

logger = logging.getLogger(__name__)

def plot_team_kmeans_clusters(colors_array: np.ndarray, labels: np.ndarray, centers: np.ndarray, output_filename: str = "team_kmeans_visualization.png"):
    """
    Generates and saves a 3D scatter plot of color clusters.

    Args:
        colors_array (np.ndarray): Array of RGB colors (N, 3).
        labels (np.ndarray): Cluster labels for each color.
        centers (np.ndarray): Cluster centers (n_clusters, 3).
        output_filename (str): Filename to save the plot.
    """
    if not (colors_array.ndim == 2 and colors_array.shape[1] == 3):
        logger.warning("Cannot plot KMeans clusters: colors_array is not 3D (RGB).")
        return
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_colors = ['blue', 'red'] 
        for i in range(len(colors_array)):
            ax.scatter(colors_array[i, 0], colors_array[i, 1], colors_array[i, 2], 
                       color=plot_colors[labels[i]], marker='o', alpha=0.6)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   marker='x', s=200, color='black', label='Team Color Centers')
        ax.set_xlabel('Red Channel')
        ax.set_ylabel('Green Channel')
        ax.set_zlabel('Blue Channel')
        ax.set_title('KMeans Clustering of Player Dominant Colors for Teams')
        plt.legend()
        plt.savefig(output_filename)
        plt.close(fig) 
        logger.info(f"KMeans team clustering visualization saved to {output_filename}")
    except Exception as e:
        logger.error(f"Error generating KMeans cluster plot: {e}", exc_info=True)

def generate_and_save_roi_report_html(html_table_rows: list[str], output_directory: str, report_filename: str = "debug_roi_processing_report.html"):
    """
    Generates an HTML report from table rows and saves it to a file.

    Args:
        html_table_rows: A list of strings, where each string is a complete <tr>...</tr> element.
        output_directory: The directory where the HTML report will be saved.
        report_filename: The name of the HTML file to be saved.
    """
    if not html_table_rows:
        logger.debug("No HTML table rows provided for the report. Skipping save.")
        return

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<title>Team ID Debug Report - ROI Processing</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
  th, td {{ border: 1px solid #ccc; padding: 10px; text-align: center; vertical-align: top; }}
  th {{ background-color: #f2f2f2; }}
  img {{ max-width: 150px; max-height: 150px; display: block; margin: 0 auto; border: 1px solid #eee; }}
  .frame-separator {{ border-top: 4px solid black !important; }}
  .frame-separator td {{ border-top: 4px solid black !important; }} /* Ensure border applies to cells too */
  .color-swatch {{ width: 40px; height: 40px; display: inline-block; border: 1px solid #666; vertical-align: middle; margin-bottom: 5px; }}
  h1 {{ text-align: center; color: #333; }}
</style>
</head>
<body>
<h1>Team Identification - Sampled ROI Details</h1>

<div style="margin-top: 20px; margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; background-color: #f9f9f9;">
  <h2>Manage Debug Information</h2>
  <button type="button" style="padding: 10px 15px; background-color: #dc3545; color: white; border: none; border-radius: 5px; cursor: pointer;">Reset Debug Information</button>
  <p style="margin-top: 15px; font-size: 0.9em; color: #555;">
    <strong>Note:</strong> For security reasons, this button cannot automatically delete files from your computer.
    To reset the debug information, please manually delete the following directories from your project's <code>LaxAI</code> folder:
  </p>
  <ul style="font-size: 0.9em; color: #555;">
    <li><code>debug_team_identification_rois/</code> (This directory, containing this report and potentially other files)</li>
    <li><code>debug_kmeans_fail_images/</code> (Contains images from failed KMeans attempts in dominant color processing)</li>
    <li><code>debug_avg_hsv_processing/</code> (Contains images from the HSV averaging dominant color processing)</li>
    <li><code>debug_histogram_hsv_processing/</code> (Contains images from the HSV histogram dominant color processing)</li>
    <li><code>debug_sampled_frames_with_detections/</code> (Contains sampled frames with detections used for initial team ID)</li>
  </ul>
  <p style="font-size: 0.9em; color: #555;">
    These directories are typically located within: <code>/Users/fernandomousinho/Documents/Learning_to_Code/LaxAI/</code>
  </p>
</div>

<p>This report shows the original ROI, the result after grass masking, the result after center cropping the masked image, and the calculated dominant color for each ROI processed during team identification sampling.</p>
<table>
  <thead>
    <tr>
      <th>Original ROI</th>
      <th>Masked (No Grass)</th>
      <th>Center Cropped (from Masked)</th>
      <th>Dominant Color (RGB)</th>
    </tr>
  </thead>
  <tbody>
    {''.join(html_table_rows)}
  </tbody>
</table>

</body>
</html>"""
    full_report_path = None # Initialize to None
    try:
        os.makedirs(output_directory, exist_ok=True)
        full_report_path = os.path.join(output_directory, report_filename)
        with open(full_report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Debug ROI processing HTML report saved to: {full_report_path}")
    except Exception as e:
        # Use output_directory if full_report_path was not successfully constructed
        log_path = full_report_path if full_report_path else output_directory
        logger.error(f"Failed to save debug HTML report to {log_path} (filename: {report_filename}): {e}", exc_info=True)

def format_roi_for_html_report(intermediate_images: Dict[str, Optional[np.ndarray]], 
                               dominant_color_rgb: Optional[np.ndarray]) -> Optional[str]:
    """
    Formats data for a single ROI into an HTML table row string.

    Args:
        intermediate_images: A dictionary containing NumPy array images for
                             'original_roi', 'masked_roi', and 'cropped_roi'.
        dominant_color_rgb: A NumPy array representing the dominant RGB color (shape (3,) or (1,3)).

    Returns:
        An HTML string for a table row (<tr>...</tr>), or None if formatting fails.
    """
    try:
        if dominant_color_rgb is None or not (dominant_color_rgb.ndim == 1 and dominant_color_rgb.shape[0] == 3) or \
                                            not (dominant_color_rgb.ndim == 2 and dominant_color_rgb.shape == (1,3)):
            # Handle cases where dominant_color_rgb might be (3,) or (1,3)
            if dominant_color_rgb is not None and dominant_color_rgb.ndim == 2 and dominant_color_rgb.shape == (1,3):
                color_to_process = dominant_color_rgb[0] # Use the first row if it's (1,3)
            elif dominant_color_rgb is not None and dominant_color_rgb.ndim == 1 and dominant_color_rgb.shape[0] == 3:
                color_to_process = dominant_color_rgb
            else:
                logger.warning(f"Invalid dominant_color_rgb for report: {dominant_color_rgb}. Skipping row.")
                return None
        else: # If it passed the initial complex check, it's likely (3,) or (1,3)
             color_to_process = dominant_color_rgb.flatten() # Ensure it's (3,)

        img_orig_b64 = VideoToools.image_to_base64_str(intermediate_images.get('original_roi'))
        img_masked_b64 = VideoToools.image_to_base64_str(intermediate_images.get('masked_roi'))
        img_cropped_b64 = VideoToools.image_to_base64_str(intermediate_images.get('cropped_roi'))

        td_orig = f'<td><img src="{img_orig_b64}" alt="Original ROI" title="Original ROI"></td>' if img_orig_b64 else "<td>N/A</td>"
        td_masked = f'<td><img src="{img_masked_b64}" alt="Masked ROI" title="Masked ROI"></td>' if img_masked_b64 else "<td>N/A</td>"
        td_cropped = f'<td><img src="{img_cropped_b64}" alt="Cropped ROI" title="Cropped ROI"></td>' if img_cropped_b64 else "<td>N/A</td>"

        r, g, b = int(color_to_process[0]), int(color_to_process[1]), int(color_to_process[2])
        color_swatch_html = f'<div class="color-swatch" style="background-color: rgb({r},{g},{b});" title="rgb({r},{g},{b})"></div> rgb({r},{g},{b})'
        td_color = f'<td>{color_swatch_html}</td>'
        
        return f"<tr>{td_orig}{td_masked}{td_cropped}{td_color}</tr>"
    except Exception as e:
        logger.error(f"Error formatting ROI for HTML report: {e}", exc_info=True)
        return None
    
# cleanup_debug_dirs.py (example, place in your project root)
import os
import shutil
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Assumes this script is in project root
LAXAI_PACKAGE_DIR = os.path.join(PROJECT_ROOT, "LaxAI")

DEBUG_DIRS_RELATIVE_TO_PACKAGE = [
    "debug_team_identification_rois",
    "debug_kmeans_fail_images",
    "debug_avg_hsv_processing",
    "debug_sampled_frames_with_detections"
]

def clean_debug_directories():
    logger.info(f"Looking for debug directories within: {LAXAI_PACKAGE_DIR}")
    for dir_name in DEBUG_DIRS_RELATIVE_TO_PACKAGE:
        dir_path = os.path.join(LAXAI_PACKAGE_DIR, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
                logger.info(f"Successfully deleted directory: {dir_path}")
            except OSError as e:
                logger.error(f"Error deleting directory {dir_path}: {e}")
        else:
            logger.info(f"Directory not found (or not a directory), skipping: {dir_path}")

if __name__ == "__main__":
    clean_debug_directories()
    logger.info("Debug directory cleanup process finished.")
