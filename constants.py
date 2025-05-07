# Application-wide constants
import os

GOOGLE_DRIVE_PATH = "/Colab_Notebooks"
DATASET_PATH = os.path.join(GOOGLE_DRIVE_PATH, "Girls-Lacrosse-8")
MODEL_NAME = "fm_rfdetr_base_model_player_only.pth"
CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.json"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".file_cache") # Cache dir in project root
CACHE_DURATION_SECONDS = 60 * 60 * 24 * 7 # 7 days