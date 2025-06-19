from typing import Optional, Literal, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

_REQUIRED_CONFIRMATIONS = 15

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

    def __init__(self, tracker_id: int) -> None:
        self.id: int = 0 #will be 0 for all unconfirmed players, and unique once confirmed
        self.tracker_id: int = tracker_id
        self.name: Optional[str] = None
        self.jersey_color: Optional[Literal["light", "dark"]] = None
        self.jersey_number: Optional[int] = None
        self._consecutive_confirmations: int = 0 
        self.is_validated: bool = False

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
    def update_or_create(cls, tracker_id: int) -> Tuple['Player', bool]:
        """
        Updates an existing player's confirmation count or creates a new player
        if one with the given tracker_id does not exist.

        Args:
            tracker_id: The tracker ID of the player to update or create.

        Returns:
            The player object and a bool indicating if it is new (True) or not (False).
        """
        if tracker_id in cls._registry:
            player = cls._registry[tracker_id]
            player._consecutive_confirmations += 1
            if player._consecutive_confirmations == cls.needed_confirmations:
                player._validate_player()
            is_new = False
            # logger.debug(f"Player tracker_id {tracker_id} has {player._consecutive_confirmations} confirmations.")

        else:
            player = cls(tracker_id=tracker_id) # Calls __init__
            player._consecutive_confirmations = 1 # First sighting counts as one confirmation
            cls._registry[tracker_id] = player
            is_new = True
            # logger.debug(f"New player created with tracker_id {tracker_id} (ID: {player.id}). Confirmations: {player._consecutive_confirmations}")

        return player, is_new
    
    @classmethod
    def get_confirmation_requirements (cls) -> int:
        """
        Returns the number of consecutive confirmations needed for a player to be validated.

        Returns:
            An integer representing the required number of confirmations.
        """
        return cls.needed_confirmations
    
    def _validate_player(self) -> None:
        self.is_validated = True
        self.id = Player._next_id
        Player._next_id += 1
