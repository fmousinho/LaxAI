import logging
logger = logging.getLogger(__name__)

from tracker.basetrack import BaseTrack

class Player(BaseTrack):
    def __init__(self):
        self.features = []
