"""
Unit tests for rendering_config.py

Tests the RenderingConfig class, StylePreset enum, and related functionality.
"""

import pytest
from shared_libs.common.rendering_config import (
    RenderingConfig,
    StylePreset,
    StyleConfig,
    get_style_config_for_preset
)


class TestStylePreset:
    """Tests for StylePreset enum."""

    def test_style_preset_values(self):
        """Test that StylePreset has expected values."""
        assert StylePreset.DEFAULT.value == "default"
        assert StylePreset.HIGHLIGHTED.value == "highlighted"
        assert StylePreset.DIMMED.value == "dimmed"
        assert StylePreset.WARNING.value == "warning"
        assert StylePreset.SUCCESS.value == "success"

    def test_style_preset_string_conversion(self):
        """Test string conversion of StylePreset."""
        assert str(StylePreset.DEFAULT) == "StylePreset.DEFAULT"
        assert StylePreset.DEFAULT.value == "default"


class TestStyleConfig:
    """Tests for StyleConfig dataclass."""

    def test_style_config_creation(self):
        """Test creating StyleConfig with default values."""
        config = StyleConfig()
        assert config.bbox_color == (0, 255, 0)  # Green default
        assert config.bbox_thickness == 2
        assert config.label_bg_color == (0, 0, 0, 255)  # Black with alpha
        assert config.label_text_color == (255, 255, 255)  # White
        assert config.label_font_size == 14

    def test_style_config_custom_values(self):
        """Test creating StyleConfig with custom values."""
        config = StyleConfig(
            bbox_color=(255, 0, 0),  # Red
            bbox_thickness=3,
            label_bg_color=(255, 255, 255, 200),  # White with alpha
            label_text_color=(0, 0, 0),  # Black
            label_font_size=16
        )
        assert config.bbox_color == (255, 0, 0)
        assert config.bbox_thickness == 3
        assert config.label_bg_color == (255, 255, 255, 200)
        assert config.label_text_color == (0, 0, 0)
        assert config.label_font_size == 16

    def test_style_config_to_dict(self):
        """Test converting StyleConfig to dictionary."""
        config = StyleConfig(bbox_color=(255, 0, 0), bbox_thickness=3)
        data = config.to_dict()

        expected = {
            "bbox_color": (255, 0, 0),
            "bbox_thickness": 3,
            "label_bg_color": (0, 0, 0, 255),
            "label_text_color": (255, 255, 255),
            "label_font_size": 14
        }
        assert data == expected

    def test_style_config_from_dict(self):
        """Test creating StyleConfig from dictionary."""
        data = {
            "bbox_color": [255, 0, 0],  # Test with list too
            "bbox_thickness": 3,
            "label_bg_color": [255, 255, 255, 200],
            "label_text_color": [0, 0, 0],
            "label_font_size": 16
        }
        config = StyleConfig.from_dict(data)

        assert config.bbox_color == (255, 0, 0)
        assert config.bbox_thickness == 3
        assert config.label_bg_color == (255, 255, 255, 200)
        assert config.label_text_color == (0, 0, 0)
        assert config.label_font_size == 16


class TestRenderingConfig:
    """Tests for RenderingConfig class."""

    def test_rendering_config_creation(self):
        """Test creating RenderingConfig with default values."""
        config = RenderingConfig()
        assert config.player_styles == {}
        assert config.tracker_styles == {}
        assert config.default_style == StylePreset.DEFAULT.value
        assert config.custom_colors == {}

    def test_rendering_config_custom_values(self):
        """Test creating RenderingConfig with custom values."""
        config = RenderingConfig(
            player_styles={1: "highlighted", 2: "dimmed"},
            tracker_styles={100: "warning"},
            default_style="success",
            custom_colors={1: (255, 0, 0), 2: (0, 255, 0)}
        )
        assert config.player_styles == {1: "highlighted", 2: "dimmed"}
        assert config.tracker_styles == {100: "warning"}
        assert config.default_style == "success"
        assert config.custom_colors == {1: (255, 0, 0), 2: (0, 255, 0)}

    def test_get_style_for_player(self):
        """Test getting style for player_id."""
        config = RenderingConfig(player_styles={1: "highlighted", 2: "dimmed"})

        assert config.get_style_for_player(1) == "highlighted"
        assert config.get_style_for_player(2) == "dimmed"
        assert config.get_style_for_player(3) == "default"  # Default when not found

    def test_get_style_for_tracker(self):
        """Test getting style for tracker_id."""
        config = RenderingConfig(tracker_styles={100: "warning", 200: "success"})

        assert config.get_style_for_tracker(100) == "warning"
        assert config.get_style_for_tracker(200) == "success"
        assert config.get_style_for_tracker(300) == "default"  # Default when not found

    def test_get_color_for_player(self):
        """Test getting custom color for player_id."""
        config = RenderingConfig(custom_colors={1: (255, 0, 0), 2: (0, 255, 0)})

        assert config.get_color_for_player(1) == (255, 0, 0)
        assert config.get_color_for_player(2) == (0, 255, 0)
        assert config.get_color_for_player(3) is None  # None when not found

    def test_set_style_for_player(self):
        """Test setting style for player_id."""
        config = RenderingConfig()
        config.set_style_for_player(1, "highlighted")
        config.set_style_for_player(2, "dimmed")

        assert config.player_styles == {1: "highlighted", 2: "dimmed"}

    def test_set_style_for_tracker(self):
        """Test setting style for tracker_id."""
        config = RenderingConfig()
        config.set_style_for_tracker(100, "warning")
        config.set_style_for_tracker(200, "success")

        assert config.tracker_styles == {100: "warning", 200: "success"}

    def test_set_color_for_player(self):
        """Test setting custom color for player_id."""
        config = RenderingConfig()
        config.set_color_for_player(1, (255, 0, 0))
        config.set_color_for_player(2, (0, 255, 0))

        assert config.custom_colors == {1: (255, 0, 0), 2: (0, 255, 0)}

    def test_update_player_mapping(self):
        """Test updating player mapping from tracker."""
        config = RenderingConfig()
        config.set_style_for_tracker(100, "highlighted")
        config.set_style_for_tracker(200, "dimmed")

        # Update tracker 100 to player 1
        config.update_player_mapping(100, 1)

        assert config.player_styles == {1: "highlighted"}
        assert config.tracker_styles == {200: "dimmed"}  # 100 should be removed

    def test_update_player_mapping_no_tracker_style(self):
        """Test updating player mapping when tracker has no style."""
        config = RenderingConfig()
        config.set_style_for_tracker(200, "dimmed")

        # Update tracker 100 (no style) to player 1
        config.update_player_mapping(100, 1)

        assert config.player_styles == {}  # No style should be added
        assert config.tracker_styles == {200: "dimmed"}

    def test_to_dict(self):
        """Test converting RenderingConfig to dictionary."""
        config = RenderingConfig(
            player_styles={1: "highlighted", 2: "dimmed"},
            tracker_styles={100: "warning"},
            default_style="success",
            custom_colors={1: (255, 0, 0), 2: (0, 255, 0)}
        )
        data = config.to_dict()

        expected = {
            "player_styles": {"1": "highlighted", "2": "dimmed"},
            "tracker_styles": {"100": "warning"},
            "default_style": "success",
            "custom_colors": {"1": [255, 0, 0], "2": [0, 255, 0]}
        }
        assert data == expected

    def test_to_json(self):
        """Test converting RenderingConfig to JSON string."""
        config = RenderingConfig(default_style="success")
        json_str = config.to_json()

        import json
        data = json.loads(json_str)
        assert data["default_style"] == "success"

    def test_from_dict(self):
        """Test creating RenderingConfig from dictionary."""
        data = {
            "player_styles": {"1": "highlighted", "2": "dimmed"},
            "tracker_styles": {"100": "warning"},
            "default_style": "success",
            "custom_colors": {"1": [255, 0, 0], "2": [0, 255, 0]}
        }
        config = RenderingConfig.from_dict(data)

        assert config.player_styles == {1: "highlighted", 2: "dimmed"}
        assert config.tracker_styles == {100: "warning"}
        assert config.default_style == "success"
        assert config.custom_colors == {1: (255, 0, 0), 2: (0, 255, 0)}

    def test_from_json(self):
        """Test creating RenderingConfig from JSON string."""
        json_str = '{"default_style": "success", "player_styles": {"1": "highlighted"}}'
        config = RenderingConfig.from_json(json_str)

        assert config.default_style == "success"
        assert config.player_styles == {1: "highlighted"}

    def test_create_default(self):
        """Test creating default RenderingConfig."""
        config = RenderingConfig.create_default()

        assert isinstance(config, RenderingConfig)
        assert config.default_style == StylePreset.DEFAULT.value
        assert config.player_styles == {}
        assert config.tracker_styles == {}
        assert config.custom_colors == {}


class TestGetStyleConfigForPreset:
    """Tests for get_style_config_for_preset function."""

    def test_get_style_config_default(self):
        """Test getting StyleConfig for default preset."""
        base_color = (255, 0, 0)  # Red
        config = get_style_config_for_preset(StylePreset.DEFAULT.value, base_color)

        assert config.bbox_color == base_color
        assert config.bbox_thickness == 2
        assert config.label_bg_color == (0, 0, 0, 255)
        assert config.label_text_color == (255, 255, 255)
        assert config.label_font_size == 14

    def test_get_style_config_highlighted(self):
        """Test getting StyleConfig for highlighted preset."""
        base_color = (0, 255, 0)  # Green
        config = get_style_config_for_preset(StylePreset.HIGHLIGHTED.value, base_color)

        assert config.bbox_color == base_color
        assert config.bbox_thickness == 3
        assert config.label_bg_color == (0, 255, 0, 255)  # base_color with alpha
        assert config.label_text_color == (255, 255, 255)
        assert config.label_font_size == 16

    def test_get_style_config_dimmed(self):
        """Test getting StyleConfig for dimmed preset."""
        base_color = (255, 255, 255)  # White
        config = get_style_config_for_preset(StylePreset.DIMMED.value, base_color)

        # Dimmed color should be base_color * 0.5
        expected_dimmed = (127, 127, 127)  # 255 * 0.5 = 127.5 -> 127 (int)
        assert config.bbox_color == expected_dimmed
        assert config.bbox_thickness == 1
        assert config.label_bg_color == (0, 0, 0, 128)  # Semi-transparent
        assert config.label_text_color == (200, 200, 200)
        assert config.label_font_size == 12

    def test_get_style_config_warning(self):
        """Test getting StyleConfig for warning preset."""
        base_color = (0, 255, 0)  # Green (ignored for warning)
        config = get_style_config_for_preset(StylePreset.WARNING.value, base_color)

        assert config.bbox_color == (255, 165, 0)  # Orange
        assert config.bbox_thickness == 3
        assert config.label_bg_color == (255, 165, 0, 255)
        assert config.label_text_color == (0, 0, 0)
        assert config.label_font_size == 14

    def test_get_style_config_success(self):
        """Test getting StyleConfig for success preset."""
        base_color = (255, 0, 0)  # Red (ignored for success)
        config = get_style_config_for_preset(StylePreset.SUCCESS.value, base_color)

        assert config.bbox_color == (0, 255, 0)  # Green
        assert config.bbox_thickness == 2
        assert config.label_bg_color == (0, 255, 0, 255)
        assert config.label_text_color == (0, 0, 0)
        assert config.label_font_size == 14

    def test_get_style_config_unknown_preset(self):
        """Test getting StyleConfig for unknown preset (defaults to default)."""
        base_color = (255, 0, 0)  # Red
        config = get_style_config_for_preset("unknown_preset", base_color)

        assert config.bbox_color == base_color
        assert config.bbox_thickness == 2
        assert config.label_bg_color == (0, 0, 0, 255)
        assert config.label_text_color == (255, 255, 255)
        assert config.label_font_size == 14

    def test_dimmed_color_tuple_validation(self):
        """Test that dimmed_color is always a 3-tuple."""
        # Test with various base colors
        test_colors = [
            (255, 255, 255),  # White
            (0, 0, 0),        # Black
            (128, 64, 192),   # Random color
        ]

        for base_color in test_colors:
            config = get_style_config_for_preset(StylePreset.DIMMED.value, base_color)

            # Verify dimmed_color is a 3-tuple
            assert isinstance(config.bbox_color, tuple)
            assert len(config.bbox_color) == 3
            assert all(isinstance(c, int) for c in config.bbox_color)

            # Verify values are correctly dimmed (multiplied by 0.5)
            expected = tuple(int(c * 0.5) for c in base_color)
            assert config.bbox_color == expected

    def test_base_color_validation(self):
        """Test that base_color must be a valid 3-tuple."""
        # Valid 3-tuple should work
        config = get_style_config_for_preset("default", (255, 0, 0))
        assert config.bbox_color == (255, 0, 0)

        # Test with invalid base_color lengths
        with pytest.raises(ValueError, match="base_color must be a tuple/list of exactly 3 integers"):
            get_style_config_for_preset("default", (255, 0))  # type: ignore # Too short

        with pytest.raises(ValueError, match="base_color must be a tuple/list of exactly 3 integers"):
            get_style_config_for_preset("default", (255, 0, 0, 0))  # type: ignore # Too long

        # Test with non-sequence
        with pytest.raises(ValueError, match="base_color must be a tuple/list of exactly 3 integers"):
            get_style_config_for_preset("default", 123)  # type: ignore # Not a sequence