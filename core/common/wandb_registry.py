import logging
import wandb
from config import wandb_config
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


REGISTRY_TYPE = 'model'
COLLECTION = 'Detections'
ALIAS = 'latest'

registry = f"wandb-registry-{REGISTRY_TYPE}/{COLLECTION}:{ALIAS}"

class WandbModelRegistry:

    def __init__(self):
        wandb.login()
        self.run = wandb.init()

    def download_blob(self, destination_blob_name, temp_checkpoint_path):
        run = wandb.init(
            entity=wandb_config.entity,
            project=wandb_config.project
        )

        artifact_name = f"wandb-registry-{REGISTRY_TYPE}/{COLLECTION}:{ALIAS}"
        # artifact_name = '<artifact_name>' # Copy and paste Full name specified on the Registry App
        fetched_artifact = run.use_artifact(artifact_or_name=artifact_name)
        download_path = fetched_artifact.download(root=temp_checkpoint_path)





        self.artifact = self.run.use_artifact('fmousinho76-home-org/wandb-registry-model/Detections:v0', type='model')
        self.artifact_dir = self.artifact.download()
