"""Server-side annotation renderer using declarative annotation schema.

This module provides native OpenCV/PIL rendering of declarative annotation
instructions, ensuring consistent visual output with client-side rendering.

Supports both legacy AnnotationRecipe format and new supervision.Detections + RenderingConfig format.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import supervision as sv

from shared_libs.common.annotation_schema import (
    AnnotationRecipe,
    AnnotationInstruction,
    AnnotationType,
    StylePreset
)
from shared_libs.common.rendering_config import RenderingConfig, StylePreset as RenderStylePreset
from shared_libs.common.player_color_palette import player_color_palette

logger = logging.getLogger(__name__)


class StyleConfig:
    """Style configuration for annotation rendering."""
    
    def __init__(
        self,
        bbox_color: Tuple[int, int, int],
        bbox_thickness: int = 2,
        label_bg_color: Tuple[int, int, int] = (0, 0, 0),
        label_text_color: Tuple[int, int, int] = (255, 255, 255),
        label_font_scale: float = 0.5,
        label_font_thickness: int = 1,
        point_radius: int = 5,
        line_thickness: int = 2
    ):
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness
        self.label_bg_color = label_bg_color
        self.label_text_color = label_text_color
        self.label_font_scale = label_font_scale
        self.label_font_thickness = label_font_thickness
        self.point_radius = point_radius
        self.line_thickness = line_thickness


class AnnotationRenderer:
    """Server-side renderer for declarative annotation instructions.
    
    This class renders annotation recipes using OpenCV, providing native
    performance while maintaining visual consistency with client-side rendering.
    """
    
    def __init__(self):
        """Initialize the renderer with color palette."""
        self.color_palette = player_color_palette
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def get_style_for_player(
        self,
        player_id: int,
        preset: StylePreset = StylePreset.DEFAULT
    ) -> StyleConfig:
        """Get style configuration for a player with given preset.
        
        Args:
            player_id: Player identifier for color lookup
            preset: Style preset to apply
            
        Returns:
            StyleConfig with appropriate colors and settings
        """
        # Get base color from palette
        color = self.color_palette.by_idx(player_id)
        base_color_bgr = color.as_bgr()  # OpenCV uses BGR
        
        # Apply preset modifications
        if preset == StylePreset.HIGHLIGHTED:
            return StyleConfig(
                bbox_color=base_color_bgr,
                bbox_thickness=3,
                label_font_scale=0.6,
                label_font_thickness=2
            )
        elif preset == StylePreset.DIMMED:
            # Reduce opacity by mixing with gray
            dimmed_color = tuple(int(c * 0.5 + 128 * 0.5) for c in base_color_bgr)
            return StyleConfig(
                bbox_color=(dimmed_color[0], dimmed_color[1], dimmed_color[2]),
                bbox_thickness=1,
                label_font_scale=0.4
            )
        elif preset == StylePreset.WARNING:
            return StyleConfig(
                bbox_color=(0, 165, 255),  # Orange in BGR
                bbox_thickness=2,
                label_bg_color=(0, 165, 255)
            )
        elif preset == StylePreset.SUCCESS:
            return StyleConfig(
                bbox_color=(0, 255, 0),  # Green in BGR
                bbox_thickness=2,
                label_bg_color=(0, 255, 0)
            )
        else:  # DEFAULT
            return StyleConfig(bbox_color=base_color_bgr)
    
    def render_bbox(
        self,
        frame: np.ndarray,
        instruction: AnnotationInstruction
    ) -> np.ndarray:
        """Render a bounding box annotation.
        
        Args:
            frame: Frame to render on (BGR format)
            instruction: Bbox instruction
            
        Returns:
            Frame with bbox rendered
        """
        style = self.get_style_for_player(
            instruction.player_id,
            instruction.style_preset
        )
        
        # Extract coordinates
        x1, y1, x2, y2 = map(int, instruction.coords)
        
        # Draw rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            style.bbox_color,
            style.bbox_thickness
        )
        
        # Draw label if text provided or use default
        label_text = instruction.label_text or f"P{instruction.player_id}"
        if instruction.confidence is not None:
            label_text += f" {instruction.confidence:.2f}"
        
        # Calculate label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label_text,
            self.font,
            style.label_font_scale,
            style.label_font_thickness
        )
        
        # Draw label background
        label_y1 = max(y1 - label_h - baseline - 5, 0)
        label_y2 = y1
        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + label_w + 5, label_y2),
            style.label_bg_color,
            -1  # Filled
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label_text,
            (x1 + 2, y1 - baseline - 2),
            self.font,
            style.label_font_scale,
            style.label_text_color,
            style.label_font_thickness
        )
        
        return frame
    
    def render_label(
        self,
        frame: np.ndarray,
        instruction: AnnotationInstruction
    ) -> np.ndarray:
        """Render a text label annotation.
        
        Args:
            frame: Frame to render on (BGR format)
            instruction: Label instruction
            
        Returns:
            Frame with label rendered
        """
        style = self.get_style_for_player(
            instruction.player_id,
            instruction.style_preset
        )
        
        x, y = map(int, instruction.coords[:2])
        label_text = instruction.label_text or f"P{instruction.player_id}"
        
        # Calculate label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label_text,
            self.font,
            style.label_font_scale,
            style.label_font_thickness
        )
        
        # Draw background
        cv2.rectangle(
            frame,
            (x, y - label_h - baseline - 2),
            (x + label_w + 4, y + 2),
            style.label_bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label_text,
            (x + 2, y),
            self.font,
            style.label_font_scale,
            style.label_text_color,
            style.label_font_thickness
        )
        
        return frame
    
    def render_point(
        self,
        frame: np.ndarray,
        instruction: AnnotationInstruction
    ) -> np.ndarray:
        """Render a point marker annotation.
        
        Args:
            frame: Frame to render on (BGR format)
            instruction: Point instruction
            
        Returns:
            Frame with point rendered
        """
        style = self.get_style_for_player(
            instruction.player_id,
            instruction.style_preset
        )
        
        x, y = map(int, instruction.coords[:2])
        
        # Draw filled circle
        cv2.circle(
            frame,
            (x, y),
            style.point_radius,
            style.bbox_color,
            -1  # Filled
        )
        
        # Draw outline
        cv2.circle(
            frame,
            (x, y),
            style.point_radius,
            (255, 255, 255),  # White outline
            1
        )
        
        return frame
    
    def render_line(
        self,
        frame: np.ndarray,
        instruction: AnnotationInstruction
    ) -> np.ndarray:
        """Render a line annotation.
        
        Args:
            frame: Frame to render on (BGR format)
            instruction: Line instruction (coords: [x1, y1, x2, y2])
            
        Returns:
            Frame with line rendered
        """
        style = self.get_style_for_player(
            instruction.player_id,
            instruction.style_preset
        )
        
        x1, y1, x2, y2 = map(int, instruction.coords[:4])
        
        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            style.bbox_color,
            style.line_thickness
        )
        
        return frame
    
    def render_recipe(
        self,
        frame_rgb: np.ndarray,
        recipe: AnnotationRecipe
    ) -> np.ndarray:
        """Render complete annotation recipe on frame.
        
        Args:
            frame_rgb: Frame in RGB format
            recipe: Annotation recipe to render
            
        Returns:
            Annotated frame in RGB format
        """
        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Render each instruction
        for instruction in recipe.instructions:
            try:
                if instruction.type == AnnotationType.BBOX:
                    frame_bgr = self.render_bbox(frame_bgr, instruction)
                elif instruction.type == AnnotationType.LABEL:
                    frame_bgr = self.render_label(frame_bgr, instruction)
                elif instruction.type == AnnotationType.POINT:
                    frame_bgr = self.render_point(frame_bgr, instruction)
                elif instruction.type == AnnotationType.LINE:
                    frame_bgr = self.render_line(frame_bgr, instruction)
                else:
                    logger.warning(f"Unknown annotation type: {instruction.type}")
            except Exception as e:
                logger.error(f"Error rendering instruction {instruction.type}: {e}")
                continue
        
        # Convert back to RGB
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    def render_frame_with_detections(
        self,
        frame_rgb: np.ndarray,
        detections: List[Dict],
        frame_id: int
    ) -> Tuple[np.ndarray, AnnotationRecipe]:
        """Render frame with detection list and return both frame and recipe.
        
        This is a convenience method that converts detection dicts to a recipe
        and renders it, useful for generating default annotations.
        
        Args:
            frame_rgb: Frame in RGB format
            detections: List of detection dictionaries
            frame_id: Frame identifier
            
        Returns:
            Tuple of (annotated frame in RGB, annotation recipe)
        """
        # Build recipe from detections
        recipe = AnnotationRecipe(frame_id=frame_id)
        
        for det in detections:
            bbox = det.get('bbox', [])
            player_id = det.get('player_id', -1)
            tracker_id = det.get('tracker_id')
            confidence = det.get('confidence')
            
            if len(bbox) == 4:
                recipe.add_bbox(
                    bbox=bbox,
                    player_id=player_id,
                    tracker_id=tracker_id,
                    confidence=confidence
                )
        
        # Render recipe
        annotated = self.render_recipe(frame_rgb, recipe)
        
        return annotated, recipe
    
    def render_detections(
        self,
        frame_rgb: np.ndarray,
        detections: sv.Detections,
        rendering_config: Optional[RenderingConfig] = None
    ) -> np.ndarray:
        """Render frame with supervision.Detections + RenderingConfig.
        
        This is the preferred method for rendering annotations using the
        supervision.Detections data structure with separate rendering configuration.
        
        Args:
            frame_rgb: Frame in RGB format
            detections: supervision.Detections object
            rendering_config: Optional rendering configuration (uses defaults if None)
            
        Returns:
            Annotated frame in RGB format
        """
        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Use default config if none provided
        if rendering_config is None:
            rendering_config = RenderingConfig()
        
        # Extract detection data
        xyxy = detections.xyxy
        confidences = detections.confidence if detections.confidence is not None else [None] * len(xyxy)
        tracker_ids = detections.tracker_id if detections.tracker_id is not None else [None] * len(xyxy)
        
        # Get player IDs from data dict
        player_ids = detections.data.get('player_id', [None] * len(xyxy)) if detections.data else [None] * len(xyxy)
        
        # Render each detection
        for i, bbox in enumerate(xyxy):
            try:
                x1, y1, x2, y2 = map(int, bbox)
                player_id = player_ids[i]
                tracker_id = tracker_ids[i]
                confidence = confidences[i]
                
                # Get style preset name
                if player_id is not None:
                    preset_name = rendering_config.get_style_for_player(player_id)
                elif tracker_id is not None:
                    preset_name = rendering_config.get_style_for_tracker(tracker_id)
                else:
                    preset_name = rendering_config.default_style
                
                # Map preset to StylePreset enum
                try:
                    preset = RenderStylePreset(preset_name)
                except ValueError:
                    preset = RenderStylePreset.DEFAULT
                
                # Check for custom color override
                custom_color = None
                if player_id is not None:
                    custom_color_rgb = rendering_config.get_color_for_player(player_id)
                    if custom_color_rgb:
                        # Convert RGB to BGR for OpenCV
                        custom_color = (custom_color_rgb[2], custom_color_rgb[1], custom_color_rgb[0])
                
                # Get color
                if custom_color:
                    color_bgr = custom_color
                else:
                    # Use palette
                    color = self.color_palette.by_idx(player_id if player_id is not None else 0)
                    color_bgr = color.as_bgr()
                
                # Map preset to thickness and font settings
                if preset == RenderStylePreset.HIGHLIGHTED:
                    thickness = 3
                    font_scale = 0.6
                elif preset == RenderStylePreset.DIMMED:
                    thickness = 1
                    font_scale = 0.4
                    # Dim the color
                    color_bgr = tuple(int(c * 0.5 + 128 * 0.5) for c in color_bgr)
                elif preset == RenderStylePreset.WARNING:
                    thickness = 2
                    font_scale = 0.5
                    color_bgr = (0, 165, 255)  # Orange in BGR
                elif preset == RenderStylePreset.SUCCESS:
                    thickness = 2
                    font_scale = 0.5
                    color_bgr = (0, 255, 0)  # Green in BGR
                else:  # DEFAULT
                    thickness = 2
                    font_scale = 0.5
                
                # Draw rectangle
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_bgr, thickness)
                
                # Build label text
                label_parts = []
                if player_id is not None:
                    label_parts.append(f"P{player_id}")
                if tracker_id is not None:
                    label_parts.append(f"T{tracker_id}")
                if confidence is not None:
                    label_parts.append(f"{confidence:.2f}")
                
                label_text = " ".join(label_parts) if label_parts else "Unknown"
                
                # Calculate label size
                (label_w, label_h), baseline = cv2.getTextSize(
                    label_text,
                    self.font,
                    font_scale,
                    1
                )
                
                # Draw label background
                label_y1 = max(y1 - label_h - baseline - 5, 0)
                label_y2 = y1
                cv2.rectangle(
                    frame_bgr,
                    (x1, label_y1),
                    (x1 + label_w + 5, label_y2),
                    color_bgr,
                    -1  # Filled
                )
                
                # Draw label text
                cv2.putText(
                    frame_bgr,
                    label_text,
                    (x1 + 2, y1 - baseline - 2),
                    self.font,
                    font_scale,
                    (255, 255, 255),  # White text
                    1
                )
                
            except Exception as e:
                logger.error(f"Error rendering detection {i}: {e}")
                continue
        
        # Convert back to RGB
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
