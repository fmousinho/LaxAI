from typing import Optional, Literal, Dict, Union, List
import logging
import numpy as np
import torch
from transformers import AutoProcessor, SiglipVisionModel
import supervision as sv


logger = logging.getLogger(__name__)

_REQUIRED_CONFIRMATIONS = 15
_MIN_HEIGHT_FOR_EMBEDDINGS = 40
_MIN_WIDTH_FOR_EMBEDDINGS = 15
_EMBEDDINGS_MODEL_PATH = "google/siglip2-base-patch16-224"

class Player:
    """
    Represents a player detected and tracked in the video.

    Attributes:
        id (int): A unique identifier for the player. It is 0 for players
                  that have not yet met the confirmation threshold. Once validated,
                  a unique positive integer is assigned.
        tracker_id (int): The ID assigned by the object tracker (e.g., ByteTrack).
        name (Optional[str]): The name of the player, if identified.
        jersey_color (Optional[Literal["light", "dark"]]): The player's jersey color.
        jersey_number (Optional[int]): The player's jersey number.
        is_validated (bool): True if the player has been confirmed enough times, False otherwise.
    """
    _next_id: int = 1
    _registry: Dict[int, 'Player'] = {} # Registry to store players by tracker_id
    needed_confirmations: int = _REQUIRED_CONFIRMATIONS
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(_EMBEDDINGS_MODEL_PATH)
    model = SiglipVisionModel.from_pretrained(_EMBEDDINGS_MODEL_PATH).to(device)

    def __init__(self, tracker_id: int) -> None:
        self.id: int = 0 #will be 0 for all unconfirmed players, and unique once confirmed
        self.tracker_id: int = tracker_id
        self.name: Optional[str] = None
        self.jersey_color: Optional[Literal["light", "dark"]] = None
        self.jersey_number: Optional[int] = None
        self.consecutive_confirmations: int = 0 
        self.is_validated: bool = False
        self.embeddings: Optional[np.ndarray] = None

    @classmethod
    def get_player_by_tid(cls, tracker_id: int) -> Optional['Player']:
        """
        Retrieves a Player instance from the registry by its tracker_id.

        Args:
            tracker_id: The tracker ID of the player to retrieve.

        Returns:
            The Player instance if found, otherwise None.
        """
        return cls._registry.get(tracker_id)

    @classmethod
    def update_or_create(cls, tracker_id: int) ->  'Player':
        """
        Updates an existing player's confirmation count or creates a new player
        if one with the given tracker_id does not exist.

        Args:
            tracker_id: The tracker ID of the player to update or create.

        Returns:
            The player object.
        """
        if tracker_id in cls._registry:
            player = cls._registry[tracker_id]
            player.consecutive_confirmations += 1
            if player.consecutive_confirmations == cls.needed_confirmations:
                player._validate_player()
            # logger.debug(f"Player tracker_id {tracker_id} has {player.consecutive_confirmations} confirmations.")

        else:
            player = cls(tracker_id=tracker_id) # Calls __init__
            player.consecutive_confirmations = 1 # First sighting counts as one confirmation
            cls._registry[tracker_id] = player
            # logger.debug(f"New player created with tracker_id {tracker_id} (ID: {player.id}). Confirmations: {player.consecutive_confirmations}")

        return player
    
    @classmethod
    def get_confirmation_requirements (cls) -> int:
        """
        Returns the number of consecutive confirmations needed for a player to be validated.

        Returns:
            An integer representing the required number of confirmations.
        """
        return cls.needed_confirmations
    
    def _validate_player(self) -> None:
        """ Changes the vadlidation status of a player and update the player sequence number. """
        self.is_validated = True
        self.id = Player._next_id
        Player._next_id += 1

    @classmethod
    def generate_embeddings_for_batch(cls, players: List['Player'], images: List[np.ndarray]) -> None:
        """
        Generates and stores image embeddings for a batch of players efficiently.

        This method processes a list of images in a single batch, which is much
        more efficient than processing them one by one. It filters out players
        whose corresponding image crops are too small and updates the `embeddings`
        attribute for each valid player.

        Args:
            players (List[Player]): A list of Player instances to update.
            images (List[np.ndarray]): A corresponding list of image crops (NumPy arrays).
        """
        if not players or not images:
            return

        if len(players) != len(images):
            logger.error(f"Batch embedding generation failed: Mismatch between players ({len(players)}) and images ({len(images)}).")
            return

        # Filter out players and images that don't meet size requirements
        valid_players_and_images = [
            (player, image) for player, image in zip(players, images)
            if image.shape[0] >= _MIN_HEIGHT_FOR_EMBEDDINGS and image.shape[1] >= _MIN_WIDTH_FOR_EMBEDDINGS
        ]

        if not valid_players_and_images:
            return

        valid_players, valid_images = zip(*valid_players_and_images)
        pil_images = [sv.cv2_to_pillow(img) for img in valid_images]

        inputs = cls.processor(images=pil_images, return_tensors="pt").to(cls.device)
        with torch.no_grad():
            outputs = cls.model(**inputs)

        batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        for i, player in enumerate(valid_players):
            player.embeddings = batch_embeddings[i]

    def generate_embeddings(self, image: np.ndarray) -> None:
        """
        Generates and stores image embeddings for the player using a SigLIP model.

        The method first checks if the input image meets minimum dimension requirements.
        If the image is too small, the method returns without generating embeddings.
        Otherwise, it processes the image using the class-level SigLIP processor and
        model to produce a feature vector. This vector is then averaged across
        the patch dimension, converted to a NumPy array, flattened, and stored in
        the `self.embeddings` attribute.
        
        Args:
            image: A NumPy array representing the player's image (e.g., a crop
                   from a video frame) in BGR or RGB format (as expected by
                   `supervision.cv2_to_pillow`).
        """
        height = image.shape[0]
        width = image.shape[1]
        if height < _MIN_HEIGHT_FOR_EMBEDDINGS or width < _MIN_WIDTH_FOR_EMBEDDINGS:
            return
        impage_pil = sv.cv2_to_pillow(image)
        inputs = self.processor(images=impage_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        md_embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        self.embeddings = np.concatenate(md_embeddings)
        
        


        
