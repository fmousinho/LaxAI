# Application-wide constants
import os

GOOGLE_DRIVE_PATH = "/Colab_Notebooks"
DATASET_PATH = os.path.join(GOOGLE_DRIVE_PATH, "Girls-Lacrosse-8")
CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.json"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".file_cache") # Cache dir in project root
CACHE_DURATION_SECONDS = 60 * 60 * 24 * 7 # 7 days

#Constants for the detection model
DEFAULT_DETECTION_MODEL = "fm_rfdetr_base_model_player_only.pth"
DEFAULT_PLAYER_CLASS_ID = 3 # Default class ID for players
DEFAULT_MODEL_DRIVE_FOLDER = "/Colab_Notebooks" # Renamed for clarity
DEFAULT_DETECTION_THRESHOLD = 0.4 # Default confidence threshold for detections


#Application constants
N_FRAMES_TO_SAMPLE = 30 # Number of frames used to train the KMeans model
MIN_DETECTIONS_FOR_SAMPLING_FRAME = 5 # Minimum detections in a frame to be considered for sampling
DEBUG_DETECTION_FRAME_SAMPLING_RATE = 50 # Log frame with detections every N frames in DEBUG mode
