import logging
from typing import Optional
import torch
import supervision as sv
from tqdm import tqdm

from ..modules.detection import DetectionModel
from ..tools.store_driver import Store

logger = logging.getLogger(__name__)


def analyse_video(
        store=Store,
        input_video=str,
        device: torch.device = torch.device("cpu"),
        start_frame=int, 
        end_frame=int
    ):
    print("hello world")
    return

