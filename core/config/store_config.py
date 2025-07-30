import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GoogleDriveConfig:
    """Configuration for Google Drive store."""
    store_driver_script_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    credentials_path: str = field(default="")
    token_path: str = field(default="")
    scopes: List[str] = field(default_factory=lambda: ['https://www.googleapis.com/auth/drive.readonly'])
    cache_dir: str = field(default="")
    cache_duration_seconds: int = 60 * 60 * 24 * 7  # Default 7 days
    
    def __post_init__(self):
        if not self.credentials_path:
            self.credentials_path = os.path.join(self.store_driver_script_dir, "../tools/credentials.json")
        if not self.token_path:
            self.token_path = os.path.join(self.store_driver_script_dir, "../tools/token.json")
        if not self.cache_dir:
            self.cache_dir = os.path.join(os.path.dirname(self.store_driver_script_dir), ".file_cache")


# Global config instance
store_config = GoogleDriveConfig()


