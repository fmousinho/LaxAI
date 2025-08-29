import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import os
import shutil
from collections import Counter
import time
import logging
from typing import Optional
from src.config.transforms import get_transforms
from src.config.all_config import model_config, clustering_config

logger = logging.getLogger(__name__)

class InferenceDataset(Dataset):
    """
    A simple dataset to load individual images for generating embeddings.
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path


class ClusteringProcessor:
    """
    A processor for clustering player embeddings using DBSCAN and reorganizing data.
    """
    
    def __init__(self, 
                 model_path: str,
                 all_crops_dir: str,
                 clustered_data_dir: str,
                 temp_dir: str,
                 embedding_dim: int = model_config.embedding_dim,
                 batch_size: int = clustering_config.batch_size,
                 initial_dbscan_eps: float = clustering_config.dbscan_eps,
                 dbscan_min_samples: int = clustering_config.dbscan_min_samples,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize the clustering processor.
        
        Args:
            model_path: Path to the trained model
            all_crops_dir: Directory containing all crop images
            clustered_data_dir: Output directory for clustered data
            embedding_dim: Dimension of the embedding vector
            batch_size: Batch size for inference
            initial_dbscan_eps: Initial DBSCAN epsilon parameter
            dbscan_min_samples: DBSCAN minimum samples parameter
            temp_dir: Temporary directory for intermediate files
            device: Device to use for inference (auto-detected if None)
        """
        self.model_path = model_path
        self.all_crops_dir = all_crops_dir
        self.clustered_data_dir = clustered_data_dir
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.dbscan_eps = initial_dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.temp_dir = temp_dir
        self.device = device
        
        # Ensure directories exist
        os.makedirs(self.all_crops_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.clustered_data_dir), exist_ok=True)

    def prepare_all_crops_directory(self, source_data_dir: str):
        """
        Copy all crops from individual tracker folders to the all_crops directory.
        
        Args:
            source_data_dir: Directory containing individual tracker folders
        """
        if not os.path.exists(source_data_dir):
            raise FileNotFoundError(f"Source data directory does not exist: {source_data_dir}")
        
        logger.info(f"Copying crops from individual tracker folders.")
        
        copied_count = 0
        for item in os.listdir(source_data_dir):
            item_path = os.path.join(source_data_dir, item)
            if os.path.isdir(item_path) and item.isdigit():  # Only process numbered tracker folders
                for crop_file in os.listdir(item_path):
                    if crop_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        src = os.path.join(item_path, crop_file)
                        dst = os.path.join(self.all_crops_dir, f"{item}_{crop_file}")  # Prefix with tracker_id
                        shutil.copy2(src, dst)
                        copied_count += 1
        
        logger.info(f"Copied {copied_count} crop images to {self.all_crops_dir}")

    def generate_all_embeddings(self, model, dataloader):
        """
        Uses the model to generate embeddings for all images in the dataloader.
        
        Args:
            model: The trained model
            dataloader: DataLoader containing the images
            
        Returns:
            Tuple of (embeddings_array, image_paths)
        """
        # Ensure model is on the correct device
        model.to(self.device)
        model.eval()
        all_embeddings = []
        all_paths = []

        logger.info(f"Generating embeddings for {len(dataloader.dataset)} images")
        start_time = time.time()
        
        with torch.no_grad():
            for i, (images, paths) in enumerate(dataloader):
                images = images.to(self.device)
                embeddings = model(images)
                all_embeddings.append(embeddings.cpu().numpy())
                all_paths.extend(paths)

        elapsed_time = time.time() - start_time
        logger.info(f"Embedding generation complete! ({elapsed_time:.2f}s, {len(all_paths)} images)")
        return np.vstack(all_embeddings), all_paths

    def cluster_embeddings(self, embeddings):
        """
        Performs DBSCAN clustering on the embedding vectors.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            Cluster labels array
        """
        logger.info(f"Starting DBSCAN clustering on {embeddings.shape[0]} embeddings")
        logger.info(f"Embedding dimensions: {embeddings.shape[1]}")
 
        logger.info(f"Parameters: eps={self.dbscan_eps:.4f}, min_samples={self.dbscan_min_samples}")        
        start_time = time.time()
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, 
                           metric='euclidean', n_jobs=-1)
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        elapsed_time = time.time() - start_time
        
        # Print clustering summary
        unique_labels = set(cluster_labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_noise = np.sum(cluster_labels == -1)
        
        logger.info(f"Clustering complete! ({elapsed_time:.2f}s)")
        logger.info(f"Found {num_clusters} unique players (clusters)")
        logger.info(f"{num_noise} images classified as noise ({num_noise/len(cluster_labels)*100:.1f}%)")
        
        # Show cluster size distribution
        if num_clusters > 0:
            cluster_counts = Counter(cluster_labels)
            if -1 in cluster_counts:
                del cluster_counts[-1]  # Remove noise count
            
            min_size = min(cluster_counts.values())
            max_size = max(cluster_counts.values())
            avg_size = np.mean(list(cluster_counts.values()))
            logger.info(f"Cluster sizes: min={min_size}, max={max_size}, avg={avg_size:.1f}")
        
        return cluster_labels

    def reorganize_data_into_clusters(self, image_paths, cluster_labels):
        """
        Copies image files into new folders named after their cluster ID.
        
        Args:
            image_paths: List of image file paths
            cluster_labels: Array of cluster labels for each image
        """
        logger.info(f"Reorganizing data into '{self.clustered_data_dir}'")
        
        if os.path.exists(self.clustered_data_dir):
            logger.info("Cleaning up existing directory...")
            shutil.rmtree(self.clustered_data_dir)
        os.makedirs(self.clustered_data_dir)
        
        # Create a mapping from image path to its cluster label
        path_to_label = dict(zip(image_paths, cluster_labels))
        
        # Count clusters and create directories
        valid_labels = [label for label in cluster_labels if label != -1]
        unique_clusters = set(valid_labels)
        
        logger.info(f"Creating {len(unique_clusters)} cluster directories")
        for label in unique_clusters:
            cluster_dir = os.path.join(self.clustered_data_dir, f"player_{label:04d}")
            os.makedirs(cluster_dir, exist_ok=True)
        
        copied_count = 0
        logger.info("Copying files to cluster directories...")
        
        for i, (path, label) in enumerate(path_to_label.items()):
            # Ignore noise points (label -1)
            if label == -1:
                continue
                
            # Create a directory for the new cluster ID if it doesn't exist
            cluster_dir = os.path.join(self.clustered_data_dir, f"player_{label:04d}")
            
            # Copy the file
            filename = os.path.basename(path)
            dst_path = os.path.join(cluster_dir, filename)
            shutil.copy2(path, dst_path)
            copied_count += 1
        
        logger.info(f"Successfully copied {copied_count} images into {len(unique_clusters)} cluster folders")
        
        # Show final statistics
        if len(unique_clusters) > 0:
            avg_per_cluster = copied_count / len(unique_clusters)
            logger.info(f"Average images per cluster: {avg_per_cluster:.1f}")

    def create_inference_transforms(self):
        """
        Create the transformation pipeline for inference.
        
        Returns:
            Composed transforms
        """
        return get_transforms('inference')

    def _setup_model_and_dataloader(self, model_class):
        """
        Setup the model and dataloader for inference (internal method).
        
        Args:
            model_class: The model class to use
            
        Returns:
            Tuple of (model, dataloader)
        """
        # Create inference transforms and dataset
        inference_transforms = self.create_inference_transforms()
        inference_dataset = InferenceDataset(image_dir=self.all_crops_dir, transform=inference_transforms)
        dataloader = DataLoader(
            inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=clustering_config.clustering_workers
        )
        
        # Load the trained model
        model = model_class(embedding_dim=self.embedding_dim)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        
        return model, dataloader

    def _count_clusters(self, cluster_labels):
        """
        Count the number of clusters (excluding noise).
        
        Args:
            cluster_labels: Array of cluster labels
            
        Returns:
            Number of clusters
        """
        unique_labels = set(cluster_labels)
        return len(unique_labels) - (1 if -1 in unique_labels else 0)

    def get_clustered_data_directory(self) -> str:
        """
        Returns the directory where all crops are stored.
        
        Returns:
            Path to the all_crops directory
        """
        return self.clustered_data_dir

    
    def find_optimal_eps(self, embeddings, target_min_clusters=None, target_max_clusters=None):
        """
        Find optimal eps value to achieve target cluster count using adaptive search.
        
        Args:
            embeddings: Numpy array of embeddings
            target_min_clusters: Minimum desired number of clusters (uses config if None)
            target_max_clusters: Maximum desired number of clusters (uses config if None)
            
        Returns:
            Tuple of (best_eps, best_cluster_labels, best_num_clusters)
        """
        
        # Use config values if not provided
        if target_min_clusters is None:
            target_min_clusters = clustering_config.target_min_clusters
        if target_max_clusters is None:
            target_max_clusters = clustering_config.target_max_clusters
        
        logger.info(f"Searching for optimal eps value to get {target_min_clusters}-{target_max_clusters} clusters")
        
        # Start with initial eps from config
        eps = clustering_config.initial_eps
        
        _MAX_EPS = clustering_config.max_eps
        _MIN_EPS = clustering_config.min_eps
        _ADJUSTMENT_FACTOR = clustering_config.eps_adjustment_factor
        _MAX_SEARCHES = clustering_config.max_eps_searches

        n_searches = 0

        while n_searches < _MAX_SEARCHES:
            n_searches += 1
            logger.info(f"Search #{n_searches}")
            self.dbscan_eps = eps
            cluster_labels = self.cluster_embeddings(embeddings)
            num_clusters = self._count_clusters(cluster_labels)
            logger.info(f"eps={eps:.4f} resulted in {num_clusters} clusters")
            if target_min_clusters <= num_clusters <= target_max_clusters:
                # In target range
                best_eps = eps
                best_cluster_labels = cluster_labels
                best_num_clusters = num_clusters
                logger.info(f"Optimal eps found!")
                break
            elif n_searches >= _MAX_SEARCHES:
                logger.warning("Reached maximum number of searches. Proceeding with last eps.")
                best_eps = eps
                best_cluster_labels = cluster_labels
                best_num_clusters = num_clusters
                break
            elif num_clusters < target_min_clusters:
                # Too few clusters, need to decrease eps
                eps = max(_MIN_EPS, self.dbscan_eps * (1 - _ADJUSTMENT_FACTOR))
            elif num_clusters > target_max_clusters:
                # Too many clusters, need to increase eps
                eps = min(_MAX_EPS, self.dbscan_eps * (1 + _ADJUSTMENT_FACTOR))
            else:
                logger.error("Unexpected condition during eps search")
        
        return best_eps, best_cluster_labels, best_num_clusters

    def process_clustering_with_search(self, model_class, source_data_dir=None, 
                                     target_min_clusters=None, target_max_clusters=None, 
                                     initial_eps=None):
        """
        Complete clustering pipeline with adaptive eps search.
        
        Args:
            model_class: The model class to use
            source_data_dir: Optional source directory to copy crops from
            target_min_clusters: Minimum desired number of clusters (uses config if None)
            target_max_clusters: Maximum desired number of clusters (uses config if None)
            initial_eps: Starting eps value (uses config if None)
            
        Returns:
            Tuple of (num_clusters, num_images_processed)
        """
        try:
            logger.info("Starting Player Re-identification Clustering Pipeline")
            logger.info(f"Using device: {self.device}")
            
            # Use config values if not provided
            if target_min_clusters is None:
                target_min_clusters = clustering_config.target_min_clusters
            if target_max_clusters is None:
                target_max_clusters = clustering_config.target_max_clusters
            if initial_eps is None:
                initial_eps = clustering_config.initial_eps
            
            # Prepare all_crops directory if needed
            if source_data_dir:
                self.prepare_all_crops_directory(source_data_dir)
            
            # Check if we have images to process
            if not os.path.exists(self.all_crops_dir) or len(os.listdir(self.all_crops_dir)) == 0:
                raise FileNotFoundError(f"No images found in '{self.all_crops_dir}'!")
            
            num_images = len([f for f in os.listdir(self.all_crops_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            logger.info(f"Found {num_images} images to process")
            
            # Setup model and dataloader
            model, dataloader = self._setup_model_and_dataloader(model_class)
            
            # Generate embeddings
            embeddings_array, image_paths = self.generate_all_embeddings(model, dataloader)
            
            # Find optimal eps value with adaptive search
            best_eps, best_cluster_labels, best_num_clusters = self.find_optimal_eps(
                embeddings_array, target_min_clusters, target_max_clusters)
            
            # Use the best clustering result
            self.dbscan_eps = best_eps
            self.reorganize_data_into_clusters(image_paths, best_cluster_labels)
            
            logger.info("All clustering steps complete!")
            logger.info(f"Dataset reorganized in '{self.clustered_data_dir}' with {best_num_clusters} clusters")
            
            return best_num_clusters, num_images
            
        except Exception as e:
            logger.error(f"Clustering pipeline failed: {str(e)}")
            raise

  