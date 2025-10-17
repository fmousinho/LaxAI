"""Rendering configuration for annotation styling.

This module provides lightweight rendering configuration that works alongside
supervision.Detections to define how annotations should be styled/colored.

The separation allows:
- supervision.Detections to remain the single source of truth for detection data
- Rendering styles to be modified without touching detection data
- Clean separation between detection logic and presentation logic
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional
from enum import Enum
import json


class StylePreset(str, Enum):
    """Predefined style presets for annotations."""
    DEFAULT = "default"
    HIGHLIGHTED = "highlighted"
    DIMMED = "dimmed"
    WARNING = "warning"
    SUCCESS = "success"


@dataclass
class StyleConfig:
    """Configuration for rendering a single annotation element.
    
    Attributes:
        bbox_color: RGB color tuple for bounding box (0-255)
        bbox_thickness: Line thickness in pixels
        label_bg_color: RGBA background color for label
        label_text_color: RGB color for label text
        label_font_size: Font size in pixels
    """
    bbox_color: Tuple[int, int, int] = (0, 255, 0)  # Green default
    bbox_thickness: int = 2
    label_bg_color: Tuple[int, int, int, int] = (0, 0, 0, 255)  # Black
    label_text_color: Tuple[int, int, int] = (255, 255, 255)  # White
    label_font_size: int = 14
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'StyleConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RenderingConfig:
    """Lightweight rendering configuration for annotations.
    
    This class maps player_id or tracker_id to rendering styles without
    modifying the underlying supervision.Detections object. It provides
    a clean separation between detection data and presentation logic.
    
    Attributes:
        player_styles: Mapping of player_id to style preset
        tracker_styles: Mapping of tracker_id to style preset
        default_style: Default style preset when no mapping exists
        custom_colors: Custom color overrides for specific player_ids
    """
    player_styles: Dict[int, str] = field(default_factory=dict)  # player_id -> preset name
    tracker_styles: Dict[int, str] = field(default_factory=dict)  # tracker_id -> preset name
    default_style: str = StylePreset.DEFAULT.value
    custom_colors: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)  # player_id -> RGB
    
    def get_style_for_player(self, player_id: int) -> str:
        """Get style preset name for player_id.
        
        Args:
            player_id: Player identifier
            
        Returns:
            Style preset name (default, highlighted, etc.)
        """
        return self.player_styles.get(player_id, self.default_style)
    
    def get_style_for_tracker(self, tracker_id: int) -> str:
        """Get style preset name for tracker_id.
        
        Args:
            tracker_id: Tracker identifier
            
        Returns:
            Style preset name (default, highlighted, etc.)
        """
        return self.tracker_styles.get(tracker_id, self.default_style)
    
    def get_color_for_player(self, player_id: int) -> Optional[Tuple[int, int, int]]:
        """Get custom color for player_id if defined.
        
        Args:
            player_id: Player identifier
            
        Returns:
            RGB color tuple or None if no custom color defined
        """
        return self.custom_colors.get(player_id)
    
    def set_style_for_player(self, player_id: int, style: str) -> None:
        """Set style preset for player_id.
        
        Args:
            player_id: Player identifier
            style: Style preset name
        """
        self.player_styles[player_id] = style
    
    def set_style_for_tracker(self, tracker_id: int, style: str) -> None:
        """Set style preset for tracker_id.
        
        Args:
            tracker_id: Tracker identifier
            style: Style preset name
        """
        self.tracker_styles[tracker_id] = style
    
    def set_color_for_player(self, player_id: int, color: Tuple[int, int, int]) -> None:
        """Set custom color for player_id.
        
        Args:
            player_id: Player identifier
            color: RGB color tuple (0-255)
        """
        self.custom_colors[player_id] = color
    
    def update_player_mapping(self, tracker_id: int, new_player_id: int) -> None:
        """Update player mapping by moving tracker's style to new player_id.
        
        This is useful when user edits change tracker_id -> player_id mappings.
        
        Args:
            tracker_id: Tracker identifier
            new_player_id: New player identifier
        """
        # If tracker had a style, move it to the new player
        if tracker_id in self.tracker_styles:
            style = self.tracker_styles.pop(tracker_id)
            self.player_styles[new_player_id] = style
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            "player_styles": {str(k): v for k, v in self.player_styles.items()},
            "tracker_styles": {str(k): v for k, v in self.tracker_styles.items()},
            "default_style": self.default_style,
            "custom_colors": {str(k): list(v) for k, v in self.custom_colors.items()}
        }
    
    def to_json(self) -> str:
        """Convert to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict) -> "RenderingConfig":
        """Create from dictionary.
        
        Args:
            data: Dictionary with rendering configuration
            
        Returns:
            RenderingConfig instance
        """
        return cls(
            player_styles={int(k): v for k, v in data.get("player_styles", {}).items()},
            tracker_styles={int(k): v for k, v in data.get("tracker_styles", {}).items()},
            default_style=data.get("default_style", StylePreset.DEFAULT.value),
            custom_colors={int(k): tuple(v) for k, v in data.get("custom_colors", {}).items()}
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "RenderingConfig":
        """Create from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            RenderingConfig instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def create_default(cls) -> "RenderingConfig":
        """Create default rendering configuration.
        
        Returns:
            RenderingConfig with default settings
        """
        return cls(
            default_style=StylePreset.DEFAULT.value
        )


def get_style_config_for_preset(preset: str, base_color: Tuple[int, int, int]) -> StyleConfig:
    """Get StyleConfig for a given preset and base color.
    
    This applies the preset rules (thickness, dimming, etc.) to a base color.
    
    Args:
        preset: Style preset name (default, highlighted, dimmed, etc.)
        base_color: Base RGB color tuple
        
    Returns:
        StyleConfig with appropriate settings
    """
    if preset == StylePreset.HIGHLIGHTED.value:
        return StyleConfig(
            bbox_color=base_color,
            bbox_thickness=3,
            label_bg_color=(*base_color, 255),
            label_text_color=(255, 255, 255),
            label_font_size=16
        )
    elif preset == StylePreset.DIMMED.value:
        # Dim the color by reducing intensity
        dimmed_color: Tuple = tuple(int(c * 0.5) for c in base_color)

        return StyleConfig(
            bbox_color=dimmed_color,
            bbox_thickness=1,
            label_bg_color=(0, 0, 0, 128),  # Semi-transparent
            label_text_color=(200, 200, 200),
            label_font_size=12
        )
    elif preset == StylePreset.WARNING.value:
        return StyleConfig(
            bbox_color=(255, 165, 0),  # Orange
            bbox_thickness=3,
            label_bg_color=(255, 165, 0, 255),
            label_text_color=(0, 0, 0),
            label_font_size=14
        )
    elif preset == StylePreset.SUCCESS.value:
        return StyleConfig(
            bbox_color=(0, 255, 0),  # Green
            bbox_thickness=2,
            label_bg_color=(0, 255, 0, 255),
            label_text_color=(0, 0, 0),
            label_font_size=14
        )
    else:  # DEFAULT
        return StyleConfig(
            bbox_color=base_color,
            bbox_thickness=2,
            label_bg_color=(0, 0, 0, 255),
            label_text_color=(255, 255, 255),
            label_font_size=14
        )
