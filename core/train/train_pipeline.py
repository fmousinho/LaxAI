import os
import logging
import traceback
from typing import Any, Dict, List, Optional


from core.common.pipeline_step import PipelineStep, StepStatus
from core.common.google_storage import GoogleStorageClient, get_storage
from core.common.pipeline import Pipeline, PipelineStatus
from core.train.dataset import LacrossePlayerDataset
from core.train.training import Training
from core.train.siamesenet import SiameseNet
from core.train.evaluator import ModelEvaluator
from config.transforms import get_transforms
from config.all_config import training_config, debug_config



logger = logging.getLogger(__name__)


class TrainPipeline(Pipeline):

    def __init__(self, tenant_id: str = "tenant1", verbose: bool = True, save_intermediate: bool = True):
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        self.verbose
        self.storage_client = get_storage(tenant_id)
        self.storage_admin = get_storage("common")
        self.model_path = training_config.model_save_path

        step_definitions = {
            "create_dataset": {
                "description": "Create dataset from crops",
                "function": self._create_dataset
            },
            "train_model": {
                "description": "Train the model",
                "function": self._train_model
            },
            "evaluate_model": {
                "description": "Evaluate the trained model",
                "function": self._evaluate_model
            }   
        }
        
        # Initialize base pipeline
        super().__init__(
            pipeline_name="training_pipeline",
            storage_client=self.storage_client,
            step_definitions=step_definitions,
            verbose=True,
            save_intermediate=False
        )

    def run(self, dataset_path: str, resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Execute the complete training pipeline for a given dataset.

        Args:
            dataset_path: Path to the training dataset directory.
            resume_from_checkpoint: Whether to check for and resume from an existing checkpoint.

        Returns:
            Dictionary with pipeline results and statistics.
        """
        if not dataset_path:
            return {"status": PipelineStatus.ERROR.value, "error": "No dataset path provided"}
        initial_context = {"dataset_path": dataset_path}
        return super().run(initial_context, resume_from_checkpoint=resume_from_checkpoint)


    def _create_dataset(self, context: dict) -> Dict[str, Any]:
        """
        Create the LacrossePlayerDataset from the given path with comprehensive validation.
        
        Args:
            context: Pipeline context containing dataset_path
            
        Returns:
            Updated context with dataset
            
        Raises:
            ValueError: If dataset path is invalid or insufficient data
            FileNotFoundError: If dataset path doesn't exist
            RuntimeError: If dataset creation fails
        """
        try:
            # Validate context parameter
            if not isinstance(context, dict):
                raise ValueError(f"Context must be a dictionary, got {type(context)}")
            
            # Extract and validate dataset path
            dataset_path = context.get("dataset_path")
            if not dataset_path:
                logger.error("Dataset path is missing from context")
                return {"status": StepStatus.ERROR.value, "error": "Dataset path is required"}
            
            # Validate dataset path type
            if not isinstance(dataset_path, str):
                logger.error(f"Dataset path must be a string, got {type(dataset_path)}")
                return {"status": StepStatus.ERROR.value, "error": f"Dataset path must be a string, got {type(dataset_path)}"}
            
            # Normalize and validate path
            dataset_path = os.path.normpath(dataset_path.strip())
            if not dataset_path or dataset_path in ['.', '..']:
                logger.error(f"Invalid dataset path: {dataset_path}")
                return {"status": StepStatus.ERROR.value, "error": f"Invalid dataset path: {dataset_path}"}
            
            # Check if dataset path exists
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset path does not exist: {dataset_path}")
                return {"status": StepStatus.ERROR.value, "error": f"Dataset path does not exist: {dataset_path}"}
            
            # Check if it's a directory
            if not os.path.isdir(dataset_path):
                logger.error(f"Dataset path is not a directory: {dataset_path}")
                return {"status": StepStatus.ERROR.value, "error": f"Dataset path is not a directory: {dataset_path}"}
            
            # Check directory permissions
            if not os.access(dataset_path, os.R_OK):
                logger.error(f"No read permission for dataset path: {dataset_path}")
                return {"status": StepStatus.ERROR.value, "error": f"No read permission for dataset path: {dataset_path}"}
            
            # Quick validation of directory structure
            subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            if len(subdirs) == 0:
                logger.error(f"No subdirectories found in dataset path: {dataset_path}")
                return {"status": StepStatus.ERROR.value, "error": f"No player directories found in dataset path: {dataset_path}"}
            
            logger.info(f"Found {len(subdirs)} potential player directories in: {dataset_path}")
            
            # Validate transforms
            try:
                transforms = get_transforms('training')
                if transforms is None:
                    logger.warning("Training transforms are None, using default transforms")
                    transforms = get_transforms('inference')  # Fallback
                    if transforms is None:
                        raise RuntimeError("Unable to load any transforms")
            except Exception as e:
                logger.error(f"Failed to load transforms: {e}")
                return {"status": StepStatus.ERROR.value, "error": f"Failed to load transforms: {str(e)}"}
            
            # Set minimum images per player with validation
            min_images_per_player = context.get("min_images_per_player", 5)
            if not isinstance(min_images_per_player, int) or min_images_per_player < 1:
                logger.warning(f"Invalid min_images_per_player: {min_images_per_player}, using default: 5")
                min_images_per_player = 5
            
            # Create the dataset with error handling
            logger.info(f"Creating dataset from: {dataset_path}")
            logger.info(f"Minimum images per player: {min_images_per_player}")
            
            try:
                dataset = LacrossePlayerDataset(
                    image_dir=dataset_path, 
                    transform=transforms,
                    min_images_per_player=min_images_per_player
                )
            except ValueError as e:
                # Handle specific dataset creation errors (e.g., insufficient players/images)
                logger.error(f"Dataset validation failed: {e}")
                return {"status": StepStatus.ERROR.value, "error": f"Dataset validation failed: {str(e)}"}
            except Exception as e:
                # Handle any other dataset creation errors
                logger.error(f"Failed to create dataset: {e}")
                return {"status": StepStatus.ERROR.value, "error": f"Failed to create dataset: {str(e)}"}
            
            # Validate created dataset
            if dataset is None:
                logger.error("Dataset creation returned None")
                return {"status": StepStatus.ERROR.value, "error": "Dataset creation failed - returned None"}
            
            # Check dataset size
            try:
                dataset_size = len(dataset)
                num_players = len(dataset.players)
                
                if dataset_size == 0:
                    logger.error("Created dataset is empty")
                    return {"status": StepStatus.ERROR.value, "error": "Created dataset is empty"}
                
                if num_players < 2:
                    logger.error(f"Dataset has insufficient players: {num_players} (minimum 2 required)")
                    return {"status": StepStatus.ERROR.value, "error": f"Dataset has insufficient players: {num_players} (minimum 2 required for training)"}
                
            except Exception as e:
                logger.error(f"Failed to validate dataset properties: {e}")
                return {"status": StepStatus.ERROR.value, "error": f"Failed to validate dataset properties: {str(e)}"}
            
            # Store dataset in context
            context['dataset'] = dataset
            context['dataset_size'] = dataset_size
            context['num_players'] = num_players
            context['min_images_per_player'] = min_images_per_player
            
            # Log success
            logger.info(f"✅ Dataset created successfully:")
            logger.info(f"   📁 Path: {dataset_path}")
            logger.info(f"   🖼️  Total images: {dataset_size}")
            logger.info(f"   👥 Players: {num_players}")
            logger.info(f"   📏 Min images per player: {min_images_per_player}")
            
            # Optionally log player statistics
            if hasattr(dataset, 'player_to_images') and len(dataset.players) <= 10:
                logger.info("   📊 Player image counts:")
                for player in dataset.players:
                    img_count = len(dataset.player_to_images[player])
                    logger.info(f"      {player}: {img_count} images")
            
            return context
            
        except Exception as e:
            # Catch any unexpected errors
            logger.error(f"Unexpected error in dataset creation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"status": StepStatus.ERROR.value, "error": f"Unexpected error: {str(e)}"}
        
    def _train_model(self, context: dict) -> Dict[str, Any]:
        """
        Train the neural network model using the Training class.
        
        Args:
            context: Pipeline context containing dataset and dataset_path
            
        Returns:
            Updated context with training results
        """
        try:
            # Validate input context
            if not isinstance(context, dict):
                error_msg = "Context must be a dictionary"
                logger.error(error_msg)
                return {"status": StepStatus.ERROR.value, "error": error_msg}
            
            # Check for required inputs
            dataset_path = context.get('dataset_path')
            if not dataset_path:
                error_msg = "dataset_path is required in context for training"
                logger.error(error_msg)
                return {"status": StepStatus.ERROR.value, "error": error_msg}
            
            dataset = context.get('dataset')
            if dataset is None:
                error_msg = "Dataset must be created before training (run create_dataset step first)"
                logger.error(error_msg)
                return {"status": StepStatus.ERROR.value, "error": error_msg}
            
            logger.info(f"Starting model training with dataset from: {dataset_path}")

            
            # Initialize Training class
            try:
                training = Training(
                    train_dir=dataset_path,
                    model_save_path=self.model_path
                )
            except Exception as e:
                logger.error(f"Failed to initialize Training class: {e}")
                return {"status": StepStatus.ERROR.value, "error": f"Failed to initialize Training class: {str(e)}"}

            # Get training transforms
            try:
                transform = get_transforms('training')
                logger.info("Loaded training transforms")
            except Exception as e:
                logger.warning(f"Failed to load training transforms: {e}, using default")
                transform = None
                return {"status": StepStatus.ERROR.value, "error": f"Failed to load training transforms: {str(e)}"}

            # Execute complete training pipeline
            logger.info("Executing training pipeline...")
            trained_model = training.train_and_save(
                model_class=SiameseNet,
                dataset_class=LacrossePlayerDataset, 
                transform=transform,
                force_pretrained=False  # Use existing weights if available
            )
            
            # Get training info for context
            training_info = training.get_training_info()
            
            logger.info("✅ Model training completed successfully")
            logger.info(f"   Model saved to: {training_info['model_save_path']}")
            logger.info(f"   Training device: {training_info['device']}")
            logger.info(f"   Embedding dimension: {training_info['embedding_dim']}")
            
            # Update context with training results
            context.update({
                'trained_model': trained_model,
                'training_info': training_info,
                'model_save_path': training_info['model_save_path'],
                'training_device': training_info['device'],
                'status': StepStatus.SUCCESS.value
            })
            
            return context
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"status": StepStatus.ERROR.value, "error": error_msg}

    def _evaluate_model(self, context: dict) -> Dict[str, Any]:
        """
        Comprehensive model evaluation using multiple methodologies.
        
        Args:
            context: Pipeline context containing trained model and dataset
            
        Returns:
            Updated context with evaluation results
        """
        try:
            # Validate input context
            if not isinstance(context, dict):
                error_msg = "Context must be a dictionary"
                logger.error(error_msg)
                return {"status": StepStatus.ERROR.value, "error": error_msg}
            
            # Check for required inputs
            trained_model = context.get('trained_model')
            if trained_model is None:
                error_msg = "Trained model must be available in context (run train_model step first)"
                logger.error(error_msg)
                return {"status": StepStatus.ERROR.value, "error": error_msg}
            
            dataset = context.get('dataset')
            if dataset is None:
                error_msg = "Dataset must be created before evaluation (run create_dataset step first)"
                logger.error(error_msg)
                return {"status": StepStatus.ERROR.value, "error": error_msg}
            
            dataset_path = context.get('dataset_path')
            if not dataset_path:
                error_msg = "dataset_path is required in context for evaluation"
                logger.error(error_msg)
                return {"status": StepStatus.ERROR.value, "error": error_msg}
            
            training_info = context.get('training_info', {})
            device = training_info.get('device', 'cpu')
            
            logger.info("Starting comprehensive model evaluation...")
            logger.info(f"Evaluation device: {device}")
            
            # Initialize evaluator
            try:
                evaluator = ModelEvaluator(
                    model=trained_model,
                    device=device,
                    threshold=0.5,  # Default similarity threshold
                    k_folds=5      # 5-fold cross-validation
                )
                logger.info("Model evaluator initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize model evaluator: {str(e)}"
                logger.error(error_msg)
                return {"status": StepStatus.ERROR.value, "error": error_msg}
            
            # Run comprehensive evaluation
            logger.info("Running comprehensive evaluation suite...")
            evaluation_results = evaluator.evaluate_comprehensive(
                dataset_path=dataset_path,
                use_validation_split=True  # Use existing train/val split
            )
            
            # Generate human-readable report
            evaluation_report = evaluator.generate_evaluation_report(evaluation_results)
            
            # Log summary to console
            logger.info("✅ Model evaluation completed successfully")
            logger.info("Evaluation Summary:")
            
            # Log key metrics
            cls_metrics = evaluation_results.get('classification_metrics', {})
            rank_metrics = evaluation_results.get('ranking_metrics', {})
            
            logger.info(f"  📊 Classification Accuracy: {cls_metrics.get('accuracy', 0):.4f}")
            logger.info(f"  📊 F1-Score: {cls_metrics.get('f1_score', 0):.4f}")
            logger.info(f"  🏆 Rank-1 Accuracy: {rank_metrics.get('rank_1_accuracy', 0):.4f}")
            logger.info(f"  🏆 Rank-5 Accuracy: {rank_metrics.get('rank_5_accuracy', 0):.4f}")
            logger.info(f"  📈 Mean Average Precision: {rank_metrics.get('mean_average_precision', 0):.4f}")
            
            # Cross-validation summary
            cv_metrics = evaluation_results.get('cross_validation', {})
            if cv_metrics:
                logger.info(f"  🔄 CV Accuracy: {cv_metrics.get('accuracy_mean', 0):.4f} ± {cv_metrics.get('accuracy_std', 0):.4f}")
                logger.info(f"  🔄 CV Rank-1: {cv_metrics.get('rank_1_accuracy_mean', 0):.4f} ± {cv_metrics.get('rank_1_accuracy_std', 0):.4f}")
            
            # Update context with evaluation results
            context.update({
                'evaluation_results': evaluation_results,
                'evaluation_report': evaluation_report,
                'evaluation_summary': {
                    'accuracy': cls_metrics.get('accuracy', 0),
                    'f1_score': cls_metrics.get('f1_score', 0),
                    'rank_1_accuracy': rank_metrics.get('rank_1_accuracy', 0),
                    'rank_5_accuracy': rank_metrics.get('rank_5_accuracy', 0),
                    'mean_average_precision': rank_metrics.get('mean_average_precision', 0)
                },
                'status': StepStatus.SUCCESS.value
            })
            
            # Print report to console if verbose
            if self.verbose:
                print("\n" + evaluation_report)
            
            return context
            
        except Exception as e:
            error_msg = f"Model evaluation failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"status": StepStatus.ERROR.value, "error": error_msg}