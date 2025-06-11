import os

class GoogleDriveConfig:
    def __init__(self, root_dir: str = "."):
        self.cache_dir = os.path.join(root_dir, ".file_cache")
        self.cache_duration = 60 * 60 * 24 * 7


