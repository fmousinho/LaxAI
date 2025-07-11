import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import numpy as np
from PIL import Image
from typing import Optional, Any, List

from .utils import log_progress
from config.transforms_config import model_config

logger = logging.getLogger(__name__)

from config.transforms_config import model_config, training_config

class EmbeddingsProcessor:
    """
    A processor class for lacrosse player re-identification model using triplet loss.
    """
    
    def __init__(self, 
                 train_dir: str,
                 embedding_dim: int = model_config.embedding_dim,
                 learning_rate: float = training_config.learning_rate,
                 batch_size: int = training_config.batch_size,
                 num_epochs: int = training_config.num_epochs,
                 margin: float = training_config.margin,
                 model_save_path: str = training_config.model_save_path,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize the embeddings processor with hyperparameters.
        
        Args:
            train_dir: Directory containing training data
            embedding_dim: Dimension of the embedding vector
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            margin: Margin for triplet loss
            model_save_path: Path to save the trained model
            device: Device to run the model on (defaults to CPU)
        """
        self.train_dir = train_dir
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.margin = margin
        self.model_save_path = model_save_path
        
        self.device = device
        
        self.model = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None
        self.dataloader = None

    def setup_data(self, dataset_class, transform: Optional[Any] = None):
        """
        Setup the dataset and dataloader. Required before training.
        
        Args:
            dataset_class: The dataset class to use (e.g., LacrossePlayerDataset)
            transform: Data transforms to apply
        """
        if not os.path.exists(self.train_dir):
            raise FileNotFoundError(f"Error: {self.train_dir} directory does not exist!")
        
        train_folders = [d for d in os.listdir(self.train_dir) 
                        if os.path.isdir(os.path.join(self.train_dir, d))]
        
        if len(train_folders) < 2:
            raise ValueError("Error: Need at least 2 player folders for triplet loss training!")
        
        logger.info(f"Found {len(train_folders)} player folders in {self.train_dir}")
        
        # Setup dataset and dataloader
        if transform is not None:
            train_dataset = dataset_class(image_dir=self.train_dir, transform=transform)
        else:
            train_dataset = dataset_class(image_dir=self.train_dir)

        n_workers = getattr(training_config, 'num_workers', 0)
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers
        )
        
        logger.info(f"Dataset size: {len(train_dataset)} images")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Number of workers: {n_workers}")
        logger.info(f"Number of batches: {len(self.dataloader)}")

    def setup_model(self, model_class, inference_only: bool = False):
        """
        Setup the model, loss function, and optimizer. Required before training.
        
        Args:
            model_class: The model class to use (e.g., SiameseNet)
            pretrained_path: Optional path to pre-trained weights to load
        """
        self.model = model_class(embedding_dim=self.embedding_dim)
        
        self.model.to(self.device)
        
        # Only setup training components if not loading pre-trained model
        if inference_only is False:
            self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            logger.info(f"Model initialized for training and moved to device: {self.device}")
        else:
            self.model.eval()
            logger.info(f"Model loaded for inference and moved to device: {self.device}")

    def train(self):
        """
        Execute the main training loop.
        
        Returns:
            The trained model
        """
        if self.model is None or self.dataloader is None:
            raise RuntimeError("Model and dataloader must be setup before training")
        
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Optimizer and loss function must be setup before training")
        
        logger.info(f"Starting training for {self.num_epochs} epochs")

        early_stopping_threshold = training_config.early_stopping_loss_ratio * self.margin
        logger.info(f"Early stopping threshold set to {early_stopping_threshold:.4f}")
        
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            batch_count = 0
            
            for i, (anchor, positive, negative, _) in enumerate(self.dataloader):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device) 
                negative = negative.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                emb_anchor, emb_positive, emb_negative = self.model.forward_triplet(
                    anchor, positive, negative)
                
                # Calculate loss
                loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
                
                # Log progress every 10 batches
                if (i + 1) % 10 == 0:
                    log_progress(logger, f"Epoch {epoch+1}/{self.num_epochs}", 
                               i + 1, len(self.dataloader), step=1)
                    logger.debug(f"Epoch {epoch+1}/{self.num_epochs}, Batch {i+1}/{len(self.dataloader)}, Current Loss: {loss.item():.4f}")

            # Calculate and log epoch summary
            epoch_loss = running_loss / batch_count if batch_count > 0 else 0.0
            logger.info(f"=== Epoch {epoch+1} Summary ===")
            logger.info(f"Average Loss: {epoch_loss:.4f}")
            logger.info("")

            # Early stopping based on config value
            
            if epoch_loss < early_stopping_threshold:
                logger.info(f"Early stopping triggered (loss {epoch_loss:.4f} < threshold {early_stopping_threshold:.4f})")
                break

        logger.info("Finished Training")
        return self.model

    def save_model(self, save_path: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            save_path: Optional custom save path. If None, uses self.model_save_path
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        
        path = save_path or self.model_save_path
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def train_and_save(self, model_class, dataset_class, transform: Optional[Any] = None):
        """
        Complete training pipeline: setup, train, and save.
        
        Args:
            model_class: The model class to use
            dataset_class: The dataset class to use  
            transform: Data transforms to apply
            
        Returns:
            The trained model
        """
        try:
            self.setup_data(dataset_class, transform)
            self.setup_model(model_class)
            trained_model = self.train()
            self.save_model()
            return trained_model
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def create_embeddings_from_crops(self,
                                    crops: List[np.ndarray], 
                                    batch_size: int = 32,
                                    transform: Optional[Any] = None) -> np.ndarray:
        """
        Creates embeddings from an array of crops using batch processing.
        
        Args:
            crops: List of crop images as numpy arrays
            batch_size: Number of crops to process in each batch
            transform: Optional transform to apply to each crop before embedding

        Returns:
            np.ndarray: Array of embeddings with shape (num_crops, embedding_dim)
        """

        if not crops:
            logger.warning("No crops provided for embedding creation")
            return np.array([])

        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        device = self.device

        # Apply transforms if provided
        if transform is not None:
            crop_shapes = [crop.shape for crop in crops if hasattr(crop, 'shape')]
            if crop_shapes and all(s == crop_shapes[0] for s in crop_shapes):
                logger.warning(f"All crops are the same size. Transform may not be used correctly")
            transformed_crops = []
            for crop in crops:
                pil_image = Image.fromarray(crop)
                transformed_tensor = transform(pil_image)
                # Convert tensor back to numpy array
                transformed_crops.append(transformed_tensor.numpy())
            crops = transformed_crops
            use_manual_normalization = False  # Transforms already handle normalization
        else:
            use_manual_normalization = True  # Need to normalize manually
            
        # Check if all crops are the same size
        

        self.model.eval()
        embeddings_list = []
        
        logger.info(f"Creating embeddings for {len(crops)} crops using batch size {batch_size}")
        
        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                # Get batch of crops
                batch_crops = crops[i:i + batch_size]
                
                # Convert to tensor and move to device
                try:
                    # Stack crops into a batch tensor
                    batch_array = np.stack(batch_crops)  # Shape: (batch_size, height, width, channels)
                    
                    # Convert to torch tensor and normalize if needed
                    batch_tensor = torch.tensor(batch_array, dtype=torch.float32, device=device)
                    
                    # If crops are in HWC format, convert to CHW format for model
                    if batch_tensor.dim() == 4 and batch_tensor.shape[-1] == 3:
                        batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                    
                    # Only normalize manually if transforms weren't applied
                    if use_manual_normalization:
                        # Normalize to [0, 1] if values are in [0, 255] range
                        if batch_tensor.max() > 1.0:
                            batch_tensor = batch_tensor / 255.0
                    
                    # Generate embeddings
                    batch_embeddings = self.model(batch_tensor)
                    
                    # Move to CPU and convert to numpy
                    batch_embeddings_np = batch_embeddings.cpu().numpy()
                    embeddings_list.append(batch_embeddings_np)
                    
                    # Log progress
                    log_progress(logger, "Processing crop batches", 
                            min(i + batch_size, len(crops)), len(crops), step=batch_size)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Create zero embeddings as fallback
                    embedding_dim = getattr(self.model, 'embedding_dim', model_config.embedding_dim)
                    zero_embeddings = np.zeros((len(batch_crops), embedding_dim), dtype=np.float32)
                    embeddings_list.append(zero_embeddings)
        
        # Concatenate all batch embeddings
        if embeddings_list:
            all_embeddings = np.concatenate(embeddings_list, axis=0)
            logger.debug(f"Successfully created {all_embeddings.shape[0]} embeddings with dimension {all_embeddings.shape[1]}")
            return all_embeddings
        else:
            logger.warning("No embeddings were created")
            return np.array([])
