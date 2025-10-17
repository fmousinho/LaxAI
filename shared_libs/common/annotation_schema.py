"""Declarative annotation schema for platform-agnostic annotation definitions.

This module provides a declarative format for defining annotations that can be
rendered natively on both client (browser Canvas) and server (OpenCV/PIL).
The focus is on defining WHAT to annotate, not HOW to render it.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict, Literal
from enum import Enum
import json


class AnnotationType(str, Enum):
    """Types of annotation instructions."""
    BBOX = "bbox"
    LABEL = "label"
    POINT = "point"
    LINE = "line"


class StylePreset(str, Enum):
    """Predefined style presets for annotations."""
    DEFAULT = "default"
    HIGHLIGHTED = "highlighted"
    DIMMED = "dimmed"
    WARNING = "warning"
    SUCCESS = "success"


@dataclass
class AnnotationInstruction:
    """Platform-agnostic annotation instruction.
    
    This represents a single annotation element (bbox, label, etc.) with
    declarative styling that can be interpreted by both client and server renderers.
    
    Attributes:
        type: Type of annotation (bbox, label, point, line)
        coords: Coordinates for the annotation (format depends on type)
        player_id: Player identifier for color/styling lookup
        tracker_id: Tracker identifier for reference
        style_preset: Predefined style name (default, highlighted, etc.)
        label_text: Optional text to display
        confidence: Optional confidence score (0.0-1.0)
        metadata: Additional custom metadata
    """
    type: AnnotationType
    coords: List[float]
    player_id: int
    tracker_id: Optional[int] = None
    style_preset: StylePreset = StylePreset.DEFAULT
    label_text: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert enums to strings
        result['type'] = self.type.value
        result['style_preset'] = self.style_preset.value
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AnnotationInstruction':
        """Create from dictionary."""
        # Convert string enums back
        data['type'] = AnnotationType(data['type'])
        data['style_preset'] = StylePreset(data.get('style_preset', 'default'))
        return cls(**data)


@dataclass
class AnnotationRecipe:
    """Collection of annotation instructions for a frame.
    
    This represents all annotations for a single frame, providing methods
    to build, serialize, and manipulate the annotation list.
    
    Attributes:
        frame_id: Frame index this recipe applies to
        instructions: List of annotation instructions
        metadata: Frame-level metadata
    """
    frame_id: int
    instructions: List[AnnotationInstruction] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_bbox(
        self,
        bbox: List[float],
        player_id: int,
        tracker_id: Optional[int] = None,
        style: StylePreset = StylePreset.DEFAULT,
        confidence: Optional[float] = None
    ) -> 'AnnotationRecipe':
        """Add a bounding box annotation.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            player_id: Player identifier
            tracker_id: Optional tracker identifier
            style: Style preset for rendering
            confidence: Optional confidence score
            
        Returns:
            Self for method chaining
        """
        instruction = AnnotationInstruction(
            type=AnnotationType.BBOX,
            coords=bbox,
            player_id=player_id,
            tracker_id=tracker_id,
            style_preset=style,
            confidence=confidence
        )
        self.instructions.append(instruction)
        return self
    
    def add_label(
        self,
        position: List[float],
        text: str,
        player_id: int,
        tracker_id: Optional[int] = None,
        style: StylePreset = StylePreset.DEFAULT
    ) -> 'AnnotationRecipe':
        """Add a text label annotation.
        
        Args:
            position: Label position [x, y]
            text: Label text
            player_id: Player identifier
            tracker_id: Optional tracker identifier
            style: Style preset for rendering
            
        Returns:
            Self for method chaining
        """
        instruction = AnnotationInstruction(
            type=AnnotationType.LABEL,
            coords=position,
            player_id=player_id,
            tracker_id=tracker_id,
            style_preset=style,
            label_text=text
        )
        self.instructions.append(instruction)
        return self
    
    def add_point(
        self,
        position: List[float],
        player_id: int,
        style: StylePreset = StylePreset.DEFAULT
    ) -> 'AnnotationRecipe':
        """Add a point marker annotation.
        
        Args:
            position: Point position [x, y]
            player_id: Player identifier
            style: Style preset for rendering
            
        Returns:
            Self for method chaining
        """
        instruction = AnnotationInstruction(
            type=AnnotationType.POINT,
            coords=position,
            player_id=player_id,
            style_preset=style
        )
        self.instructions.append(instruction)
        return self
    
    def update_player_id(self, tracker_id: int, new_player_id: int) -> bool:
        """Update player_id for all instructions with given tracker_id.
        
        This is the primary method for handling user edits in the browser.
        
        Args:
            tracker_id: Tracker ID to match
            new_player_id: New player ID to assign
            
        Returns:
            True if any instructions were updated
        """
        updated = False
        for instruction in self.instructions:
            if instruction.tracker_id == tracker_id:
                instruction.player_id = new_player_id
                updated = True
        return updated
    
    def remove_by_tracker(self, tracker_id: int) -> int:
        """Remove all instructions with given tracker_id.
        
        Args:
            tracker_id: Tracker ID to remove
            
        Returns:
            Number of instructions removed
        """
        original_count = len(self.instructions)
        self.instructions = [
            inst for inst in self.instructions 
            if inst.tracker_id != tracker_id
        ]
        return original_count - len(self.instructions)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'frame_id': self.frame_id,
            'instructions': [inst.to_dict() for inst in self.instructions],
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AnnotationRecipe':
        """Create from dictionary."""
        instructions = [
            AnnotationInstruction.from_dict(inst_data)
            for inst_data in data.get('instructions', [])
        ]
        return cls(
            frame_id=data['frame_id'],
            instructions=instructions,
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AnnotationRecipe':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class VideoAnnotationRecipes:
    """Collection of annotation recipes for multiple frames.
    
    This represents all user-edited annotations for a video generation request.
    
    Attributes:
        session_id: Video session identifier
        video_id: Video identifier
        frame_recipes: Dictionary mapping frame_id to AnnotationRecipe
        generation_settings: Settings for video generation
    """
    session_id: str
    video_id: str
    frame_recipes: Dict[int, AnnotationRecipe] = field(default_factory=dict)
    generation_settings: Dict = field(default_factory=dict)
    
    def add_frame_recipe(self, recipe: AnnotationRecipe) -> None:
        """Add or update recipe for a frame.
        
        Args:
            recipe: Annotation recipe to add
        """
        self.frame_recipes[recipe.frame_id] = recipe
    
    def get_frame_recipe(self, frame_id: int) -> Optional[AnnotationRecipe]:
        """Get recipe for a specific frame.
        
        Args:
            frame_id: Frame index
            
        Returns:
            AnnotationRecipe if found, None otherwise
        """
        return self.frame_recipes.get(frame_id)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'video_id': self.video_id,
            'frame_recipes': {
                str(frame_id): recipe.to_dict()
                for frame_id, recipe in self.frame_recipes.items()
            },
            'generation_settings': self.generation_settings
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VideoAnnotationRecipes':
        """Create from dictionary."""
        frame_recipes = {
            int(frame_id): AnnotationRecipe.from_dict(recipe_data)
            for frame_id, recipe_data in data.get('frame_recipes', {}).items()
        }
        return cls(
            session_id=data['session_id'],
            video_id=data['video_id'],
            frame_recipes=frame_recipes,
            generation_settings=data.get('generation_settings', {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'VideoAnnotationRecipes':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
