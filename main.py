# Standard Library
import argparse
import logging
import os
import sys

# Third-party
import torch
from dotenv import load_dotenv

# Local Application/Library
from . import application as app
from .config import logging_config
from .tools import utils
from .tools.store_driver import Store
from .analyse import analyse as analyse_module


logger = logging.getLogger(__name__)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DOTENV_PATH = os.path.join(PACKAGE_DIR, ".env")

DEFAULT_TORCH_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else \
    "mps" if torch.backends.mps.is_available() else \
    "cpu"
)

def main() -> int:
    logger.info("---------- LaxAI Starting Application ----------")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="LaxAI Video Processing Application.")
    parser.add_argument("input_video",
                        type=str,
                        help="Local file path of the video to be processed.")
    parser.add_argument("-o", "--output_video_path",
                        type=str,
                        default="results.mp4",
                        help="Path for the output processed video. Default: results.mp4")
    parser.add_argument("--device",
                        type=str,
                        default=DEFAULT_TORCH_DEVICE.type,
                        choices=['cuda', 'mps', 'cpu'],
                        help=f"Computation device. Default: auto-detect ({DEFAULT_TORCH_DEVICE.type})")
    parser.add_argument("--debug_frames",
                        type=int,
                        default=None,
                        metavar='N',
                        help="Process only the first N frames for debugging. Default: process all frames.")
    parser.add_argument("--log_level",
                        type=str,
                        default="INFO",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level. Default: INFO")
    parser.add_argument("--analyse",
                        type=utils.frame_interval_type,
                        default=None, # Explicitly set default to None
                        metavar='START:END',
                        help="Analyse a specific frame interval (e.g., '100:500'). This runs in a special analysis mode.")
    parser.add_argument("--report",
                        action='store_true',
                        help="Generate a player analysis HTML report at the end of the main application run.")

    args = parser.parse_args()

    # --- Configure Logging Level based on args ---
    try:
        logging.getLogger('LaxAI').setLevel(args.log_level.upper())
    except Exception as e:
        # Fallback if the above fails, log at the initial level
        logger.error(f"Failed to dynamically set log level to {args.log_level.upper()}: {e}")

    # --- Environment and System Checks ---
    selected_device = torch.device(args.device)
    logger.info(f"Using device: {selected_device}")

    if not utils.check_requirements():
        logger.error("Requirements check failed. Please install missing packages.")
        return 1 # Exit with error code

    # --- Load .env ---
    if os.path.exists(_DOTENV_PATH):
        if load_dotenv(dotenv_path=_DOTENV_PATH, verbose=True): # verbose=True logs INFO messages from dotenv
            logger.info(f".env file loaded successfully from {_DOTENV_PATH}.")
        else:
            logger.warning(f".env file at {_DOTENV_PATH} was found but may be empty or failed to load variables.")
    else:
        logger.info(f".env file not found at {_DOTENV_PATH}. Proceeding with environment variables or defaults.")

    # --- Verify if video file provided exists  ---
    input_video = args.input_video
    if not os.path.exists(input_video):
        logger.error(f"Input video file '{input_video}' does not exist. Please provide a valid file path.")
        return 1

    # --- Main Application Logic ---
    try:
        with Store() as store:
            # Store is crucial if downloading video or models from the store object
            if not store.is_initialized():
                logger.critical("Store initialization failed. Exiting.")
                return 1
            
            #Analyse is a mode to evalue details of ByteTracker implementation. Can be verbose.
            if args.analyse:
                start_frame, end_frame = args.analyse
                logger.warning(f"Entering analysis mode. Check directory for results.")
                analyse_module.analyse_video (
                    store=store,
                    input_video=input_video,
                    start_frame=start_frame, 
                    end_frame=end_frame,
                    device=selected_device
                    )
                logger.info("Analysis completed successfully.")
                return 0
            
            app.run_application(
                store=store,
                input_video=input_video,
                output_video_path=args.output_video_path,
                device=selected_device,
                debug_max_frames=args.debug_frames,
                #generate_report=args.report
            )
        logger.info("Application run completed successfully.")
        return 0

    except Exception as e:
        logger.critical(f"An unhandled exception occurred in the main application: {e}", exc_info=True)
        return 1
    finally:
        logger.info("---------- LaxAI Application Finished ----------")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
