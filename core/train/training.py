import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Optional, Any

from config.all_config import model_config, training_config

logger = logging.getLogger(__name__)


class Training:
    """
    A training class specifically for neural network training using triplet loss.
    Focused on training lacrosse player re-identification models.
    """
    
    def __init__(self, 
                 train_dir: str,
                 embedding_dim: int = model_config.embedding_dim,
                 learning_rate: float = training_config.learning_rate,
                 batch_size: int = training_config.batch_size,
                 num_epochs: int = training_config.num_epochs,
                 margin: float = training_config.margin,
                 model_save_path: str = training_config.model_save_path,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')):
        """
        Initialize the training class with hyperparameters.
        
        Args:
            train_dir: Directory containing training data
            embedding_dim: Dimension of the embedding vector
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            margin: Margin for triplet loss
            model_save_path: Path to save the trained model
            device: Device to run the model on (CPU, GPU, or MPS)
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
        
        logger.info(f"Training initialized with device: {self.device}")
        logger.info(f"Training directory: {self.train_dir}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Number of epochs: {self.num_epochs}")
        logger.info(f"Triplet margin: {self.margin}")

    def setup_data(self, dataset_class, transform: Optional[Any] = None):
        """
        Setup the dataset and dataloader for training.
        
        Args:
            dataset_class: The dataset class to use (e.g., LacrossePlayerDataset)
            transform: Data transforms to apply
            
        Raises:
            FileNotFoundError: If training directory doesn't exist
            ValueError: If insufficient player folders for training
        """
        if not os.path.exists(self.train_dir):
            raise FileNotFoundError(f"Training directory does not exist: {self.train_dir}")
        
        train_folders = [d for d in os.listdir(self.train_dir) 
                        if os.path.isdir(os.path.join(self.train_dir, d))]
        
        if len(train_folders) < 2:
            raise ValueError("Need at least 2 player folders for triplet loss training!")
        
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
        
        logger.info(f"Dataset setup complete:")
        logger.info(f"  Dataset size: {len(train_dataset)} images")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Number of workers: {n_workers}")
        logger.info(f"  Number of batches: {len(self.dataloader)}")

    def setup_model(self, model_class, force_pretrained: bool = False):
        """
        Setup the model, loss function, and optimizer for training.
        
        Args:
            model_class: The model class to use (e.g., SiameseNet)
            force_pretrained: If True, ignore saved weights and start with pre-trained backbone
            
        Raises:
            RuntimeError: If model setup fails
        """
        try:
            self.model = model_class(embedding_dim=self.embedding_dim)
            
            # Load existing weights if available and not forcing pre-trained
            if os.path.exists(self.model_save_path) and not force_pretrained:
                try:
                    self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
                    logger.info(f"✓ Loaded existing fine-tuned weights from {self.model_save_path}")
                    print(f"✓ Continuing training from existing model: {self.model_save_path}")
                except Exception as e:
                    logger.warning(f"Failed to load existing weights from {self.model_save_path}: {e}")
                    logger.info("Falling back to pre-trained ResNet18 weights")
                    print(f"⚠ Could not load saved weights, using pre-trained ResNet18: {e}")
            else:
                if force_pretrained:
                    logger.info(f"Forcing fresh start with pre-trained ResNet18 weights")
                    print(f"🔄 Starting fresh with pre-trained ResNet18 backbone")
                else:
                    logger.info(f"No existing model found at {self.model_save_path}, using pre-trained ResNet18 weights")
                    print(f"🆕 Starting training with pre-trained ResNet18 backbone")
            
            self.model.to(self.device)
            
            # Setup training components
            self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=getattr(training_config, 'weight_decay', 1e-4)
            )
            
            logger.info(f"Model setup complete and moved to device: {self.device}")
            logger.info(f"Loss function: TripletMarginLoss (margin={self.margin})")
            logger.info(f"Optimizer: Adam (lr={self.learning_rate}, weight_decay={getattr(training_config, 'weight_decay', 1e-4)})")
            
        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            raise RuntimeError(f"Failed to setup model: {e}")

    def train(self):
        """
        Execute the main training loop with early stopping.
        
        Returns:
            The trained model
            
        Raises:
            RuntimeError: If required components are not setup
        """
        if self.model is None or self.dataloader is None:
            raise RuntimeError("Model and dataloader must be setup before training")
        
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Optimizer and loss function must be setup before training")
        
        logger.info(f"Starting training for {self.num_epochs} epochs")

        # Early stopping configuration
        early_stopping_threshold = training_config.early_stopping_loss_ratio * self.margin
        early_stopping_patience = getattr(training_config, 'early_stopping_patience', None)
        
        logger.info(f"Early stopping threshold set to {early_stopping_threshold:.4f}")
        if early_stopping_patience is not None:
            logger.info(f"Early stopping patience set to {early_stopping_patience} epochs")
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        
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

            # Calculate and log epoch summary
            epoch_loss = running_loss / batch_count if batch_count > 0 else 0.0
            logger.info(f"=== Epoch {epoch+1}/{self.num_epochs} Summary ===")
            logger.info(f"Average Loss: {epoch_loss:.4f}")
            
            # Early stopping based on loss threshold
            if epoch_loss < early_stopping_threshold:
                logger.info(f"Early stopping triggered (loss {epoch_loss:.4f} < threshold {early_stopping_threshold:.4f})")
                break
            
            # Early stopping based on patience
            if early_stopping_patience is not None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    logger.debug(f"New best loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.debug(f"Patience counter: {patience_counter}/{early_stopping_patience}")
                    
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered due to patience ({patience_counter} epochs without improvement)")
                    break

        logger.info("Training completed successfully")
        return self.model

    def save_model(self, save_path: Optional[str] = None):
        """
        Save the trained model weights.
        
        Args:
            save_path: Optional custom save path. If None, uses self.model_save_path
            
        Raises:
            RuntimeError: If no model to save
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        
        path = save_path or self.model_save_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model weights saved to {path}")
        print(f"✓ Model saved to: {path}")

    def train_and_save(self, model_class, dataset_class, transform: Optional[Any] = None, force_pretrained: bool = False):
        """
        Complete training pipeline: setup data, setup model, train, and save.
        
        Args:
            model_class: The model class to use (e.g., SiameseNet)
            dataset_class: The dataset class to use (e.g., LacrossePlayerDataset)
            transform: Data transforms to apply
            force_pretrained: If True, ignore saved weights and start fresh
            
        Returns:
            The trained model
            
        Raises:
            Exception: If any step in the training pipeline fails
        """
        try:
            logger.info("Starting complete training pipeline")
            
            # Setup data
            logger.info("Setting up training data...")
            self.setup_data(dataset_class, transform)
            
            # Setup model
            logger.info("Setting up model...")
            self.setup_model(model_class, force_pretrained)
            
            # Train
            logger.info("Starting training...")
            trained_model = self.train()
            
            # Save
            logger.info("Saving model...")
            self.save_model()
            
            logger.info("Training pipeline completed successfully")
            print(f"🎉 Training completed successfully!")
            
            return trained_model
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            print(f"❌ Training failed: {str(e)}")
            raise

    def get_training_info(self):
        """
        Get information about the current training setup.
        
        Returns:
            Dictionary with training configuration details
        """
        return {
            'train_dir': self.train_dir,
            'embedding_dim': self.embedding_dim,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'margin': self.margin,
            'model_save_path': self.model_save_path,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'dataloader_ready': self.dataloader is not None,
            'optimizer_ready': self.optimizer is not None,
            'loss_fn_ready': self.loss_fn is not None
        }
