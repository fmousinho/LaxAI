"""Pub/Sub configuration management."""

import os
from pydantic import BaseModel, Field


class PubSubConfig(BaseModel):
    """Pub/Sub configuration.
    
    Reads from environment variables with sensible defaults.
    """
    project_id: str = Field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID", "laxai-466119"),
        description="GCP project ID"
    )
    training_topic: str = Field(
        default_factory=lambda: os.getenv("PUBSUB_TRAINING_TOPIC", "training-jobs"),
        description="Training jobs Pub/Sub topic name"
    )
    
    def get_topic_path(self, publisher_client) -> str:
        """Get full topic path for publishing.
        
        Args:
            publisher_client: Pub/Sub publisher client
            
        Returns:
            Full topic path string
        """
        return publisher_client.topic_path(self.project_id, self.training_topic)
