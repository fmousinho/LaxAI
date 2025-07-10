import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import numpy as np
from .utils import log_progress
from typing import Optional, Any, List

logger = logging.getLogger(__name__)


class Trainer:
    """
    A trainer class for lacrosse player re-identification model using triplet loss.
    """
    
    def __init__(self, 
                 train_dir: str,
                 embedding_dim: int = 128,
                 learning_rate: float = 0.001,
                 batch_size: int = 16,
                 num_epochs: int = 5,
                 margin: float = 0.5,
                 model_save_path: str = 'lacrosse_reid_model.pth'):
        """
        Initialize the trainer with hyperparameters.
        
        Args:
            train_dir: Directory containing training data
            embedding_dim: Dimension of the embedding vector
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            margin: Margin for triplet loss
            model_save_path: Path to save the trained model
        """
        self.train_dir = train_dir
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.margin = margin
        self.model_save_path = model_save_path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        
        self.model = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None
        self.dataloader = None

    def setup_data(self, dataset_class, transform: Optional[Any] = None):
        """
        Setup the dataset and dataloader.
        
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
        self.dataloader = DataLoader(train_dataset, 
                                   batch_size=self.batch_size, 
                                   shuffle=True, 
                                   num_workers=0)
        
        logger.info(f"Dataset size: {len(train_dataset)} images")
        logger.info(f"Number of batches: {len(self.dataloader)}")

    def setup_model(self, model_class):
        """
        Setup the model, loss function, and optimizer.
        
        Args:
            model_class: The model class to use (e.g., SiameseNet)
        """
        self.model = model_class(embedding_dim=self.embedding_dim)
        self.model.to(self.device)
        
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        logger.info(f"Model initialized and moved to device: {self.device}")

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
                    logger.info(f"Current Loss: {loss.item():.4f}")

            # Calculate and log epoch summary
            epoch_loss = running_loss / batch_count if batch_count > 0 else 0.0
            logger.info(f"=== Epoch {epoch+1} Summary ===")
            logger.info(f"Average Loss: {epoch_loss:.4f}")
            logger.info("")

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
                                    device: Optional[torch.device] = None) -> np.ndarray:
        """
        Creates embeddings from an array of crops using batch processing.
        
        Args:
            crops: List of crop images as numpy arrays
            model: The model to use for generating embeddings
            batch_size: Number of crops to process in each batch
            device: Device to run the model on (defaults to model's device)
            
        Returns:
            np.ndarray: Array of embeddings with shape (num_crops, embedding_dim)
        """
        if not crops:
            logger.warning("No crops provided for embedding creation")
            return np.array([])


        if self.model is None:
            raise RuntimeError("Model not initialized")
            
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
                    embedding_dim = getattr(self.model, 'embedding_dim', 128)
                    zero_embeddings = np.zeros((len(batch_crops), embedding_dim), dtype=np.float32)
                    embeddings_list.append(zero_embeddings)
        
        # Concatenate all batch embeddings
        if embeddings_list:
            all_embeddings = np.concatenate(embeddings_list, axis=0)
            logger.info(f"Successfully created {all_embeddings.shape[0]} embeddings with dimension {all_embeddings.shape[1]}")
            return all_embeddings
        else:
            logger.warning("No embeddings were created")
            return np.array([])




