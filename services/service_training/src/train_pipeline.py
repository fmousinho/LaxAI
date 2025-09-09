import importlib
import logging
import os
import traceback
from typing import Any, Callable, Dict, List, Optional

from dataset import LacrossePlayerDataset
from evaluator import Evaluator, ModelEvaluator
from training_loop import Training
from transforms import get_transforms
from wandb_logger import wandb_logger

from shared_libs.common.google_storage import GCSPaths, get_storage
from shared_libs.common.pipeline import Pipeline, PipelineStatus
from shared_libs.common.pipeline_step import StepStatus
from shared_libs.config.all_config import (model_config, training_config,
                                           wandb_config)

logger = logging.getLogger(__name__)


class TrainPipeline(Pipeline):

    def __init__(self, tenant_id: str = "tenant1", verbose: bool = True, save_intermediate: bool = True, pipeline_name: str = "training_pipeline", **training_kwargs):
        """
        Initialize the training pipeline.
        """
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        self.storage_client = get_storage(tenant_id)
       

        if training_kwargs.get("model_class_module"):
            model_class_module = training_kwargs["model_class_module"]
        else:
            model_class_module = model_config.model_class_module
        if training_kwargs.get("model_class_str"):
            model_class_str = training_kwargs["model_class_str"]
        else:
            model_class_str = model_config.model_class_str
        module = importlib.import_module(model_class_module)

        self.collection_name = model_class_module
        
        self.model_class = getattr(module, model_class_str)
        self.path_manager = GCSPaths()
        self.training_kwargs = training_kwargs


        step_definitions = {
            "create_dataset": {
                "description": "Create dataset from crops",
                # Resolve method at call time to allow instance-level patching in tests.
                # Normalize legacy/mocked return values (some tests return (train_ds, val_ds)).
                "function": (lambda context, _self=self: (
                    # Call the bound method (may be patched in tests)
                    (lambda _res: (
                        # If a tuple/list of length 2 is returned, convert to expected context
                        {'training_dataset': _res[0], 'validation_dataset': _res[1], 'dataset_mode': 'single', 'status': StepStatus.COMPLETED.value}
                        if isinstance(_res, (tuple, list)) and len(_res) == 2 else _res
                    ))(getattr(_self, '_create_dataset')(context))
                ))
            },
            "train_model": {
                "description": "Train the model",
                "function": (lambda context, _self=self, stop_callback=None: getattr(_self, '_train_model')(context, stop_callback=stop_callback))
            },
            "evaluate_model": {
                "description": "Evaluate the trained model",
                "function": (lambda context, _self=self, stop_callback=None: getattr(_self, '_evaluate_model')(context, stop_callback=stop_callback))
            }   
        }
        
        # Initialize base pipeline using the provided pipeline_name so external
        # callers (for example the API) can create uniquely-named pipelines
        # (e.g. training_pipeline_<task_id>) and cancel them by that name.
        super().__init__(
            pipeline_name=pipeline_name,
            storage_client=self.storage_client,
            step_definitions=step_definitions,
            verbose=verbose,
            save_intermediate=save_intermediate
        )

    def run(self, dataset_name: str | List[str], resume_from_checkpoint: bool = True, wandb_run_tags: Optional[List[str]] = None, custom_name: str = "run") -> Dict[str, Any]:
        """
        Execute the complete training pipeline for a given dataset.

        Args:
            dataset_name: Dataset name(s) - either a single string or list of strings for multi-dataset training.
            resume_from_checkpoint: Whether to check for and resume from an existing wandb checkpoint artifact.
            wandb_run_tags: Optional tags for the wandb run.
            custom_name: Custom name for the training run (used in wandb and logging).

        Returns:
            Dictionary with pipeline results and statistics.
        """
        if wandb_config.enabled:
            # Initialize WandB with comprehensive configuration for the entire pipeline
            config = {
                "pipeline": "training_pipeline",
                "custom_name": custom_name,
                "resume_from_checkpoint": resume_from_checkpoint,
                # Training parameters (will be available for both training and evaluation)
                "learning_rate": self.training_kwargs.get('learning_rate', getattr(training_config, 'learning_rate', 0.001)),
                "batch_size": self.training_kwargs.get('batch_size', getattr(training_config, 'batch_size', 32)),
                "num_epochs": self.training_kwargs.get('num_epochs', getattr(training_config, 'num_epochs', 50)),
                "margin": self.training_kwargs.get('margin', getattr(training_config, 'margin', 1.0)),
                "weight_decay": self.training_kwargs.get('weight_decay', getattr(training_config, 'weight_decay', 0.0001)),
                "num_workers": self.training_kwargs.get('num_workers', getattr(training_config, 'num_workers', 4)),
                "model_class": self.model_class.__name__,
                "dataset_name": dataset_name
            }

            wandb_logger.init_run(config=config, run_name=f"{custom_name}", tags=wandb_run_tags)
        # On macOS / some CI environments torch DataLoader uses 'spawn' by default
        # which can fail when tests run code that starts worker processes from
        # imported modules. Prefer 'fork' where available for compatibility.
        try:
            import multiprocessing as mp
            if os.name == 'posix' and 'darwin' in os.uname().sysname.lower():
                try:
                    mp.set_start_method('fork')
                    logger.debug("Set multiprocessing start method to 'fork' for macOS compatibility")
                except RuntimeError:
                    # start method already set; ignore
                    pass
        except Exception:
            # If we can't set it, continue; tests will handle num_workers=0 fallback
            logger.debug("Could not set multiprocessing start method; continuing")
        try:
            if not dataset_name:
                return {"status": PipelineStatus.ERROR.value, "error": "No dataset name provided"}
            context = {
                "dataset_name": dataset_name, 
                "custom_name": custom_name, 
                "resume_from_checkpoint": resume_from_checkpoint
            }
            # Pass to generic pipeline without resume_from_checkpoint for the pipeline steps
            results = super().run(context)
            
            # Add wandb run ID to results for test verification
            if wandb_logger.run and hasattr(wandb_logger.run, 'id'):
                results["run_id"] = wandb_logger.run.id
            
            wandb_logger.finish()

            # If the pipeline reported step failures, surface specific errors
            if isinstance(results, dict) and results.get('steps_failed', 0) > 0:
                errors = results.get('errors', []) or []
                # If any error message mentions insufficient players, raise ValueError
                for err in errors:
                    if isinstance(err, str) and 'Insufficient players' in err:
                        raise ValueError(err)
                # Otherwise raise a generic RuntimeError to surface failure to callers/tests
                raise RuntimeError(f"Pipeline failed with errors: {errors}")

            return results
        except Exception as e:
            logger.error(f"Error occurred during training pipeline run: {e}")
            wandb_logger.finish()
            # Re-raise to allow callers/tests to detect specific errors
            raise


    def _create_dataset(self, context: dict) -> Dict[str, Any]:
        """
        Create the LacrossePlayerDataset from the given path(s) with comprehensive validation.
        
        Supports both single dataset and multi-dataset modes:
        - Single dataset: dataset_name is a string, creates datasets from train/val folders
        - Multi-dataset: dataset_name is a list, creates datasets from multiple train/val folders
        
        Args:
            context: Pipeline context containing dataset_name (str or List[str])
            
        Returns:
            Updated context with dataset
            
        Raises:
            ValueError: If dataset path is invalid or insufficient data
            FileNotFoundError: If dataset path doesn't exist
            RuntimeError: If dataset creation fails
        """
        try:
            # Extract and validate dataset name
            dataset_name = context.get("dataset_name")
            if not dataset_name:
                logger.error("Dataset name is missing from context")
                return {"status": StepStatus.ERROR.value, "error": "Dataset name is required"}

            # Handle both single dataset and multi-dataset modes
            if isinstance(dataset_name, str):
                # Single dataset mode
                try:
                    train_folder = self.path_manager.get_path("train_dataset", dataset_id=dataset_name)
                    val_folder = self.path_manager.get_path("val_dataset", dataset_id=dataset_name)
                    train_folders = [train_folder]  # Convert to list for consistency
                    val_folders = [val_folder]      # Convert to list for consistency
                    dataset_mode = "single"
                    logger.info(f"🔄 Creating single dataset from: {dataset_name}")
                except KeyError as e:
                    logger.error(f"Could not find train and val folders for {dataset_name}: {e}")
                    return {"status": StepStatus.ERROR.value, "error": str(e)}
                
            elif isinstance(dataset_name, list):
                # Multi-dataset mode
                train_folders = []
                val_folders = []
                for name in dataset_name:
                    try:
                        train_folder = self.path_manager.get_path("train_dataset", dataset_id=name)
                        val_folder = self.path_manager.get_path("val_dataset", dataset_id=name)
                        train_folders.append(train_folder)
                        val_folders.append(val_folder)
                    except KeyError as e:
                        logger.warning(f"Skipping dataset {name} due to missing folders")
                        continue
                if len(train_folders) == 0:
                    logger.error(f"No valid train folders found for {dataset_name}")
                    return {"status": StepStatus.ERROR.value, "error": f"No valid train folders found for {dataset_name}"}
                if len(val_folders) == 0:
                    logger.warning(f"No valid validation folders found for {dataset_name}")
                dataset_mode = "multi"
                logger.info(f"🔄 Creating multi-dataset from {len(dataset_name)} datasets: {dataset_name}")
            
            else:
                raise ValueError(f"dataset_name must be str or List[str], got {type(dataset_name)}")
            
            if not train_folders or not val_folders:
                logger.error(f"No valid train or validation folders found for {dataset_name}")
                return {"status": StepStatus.ERROR.value, "error": f"No valid train or validation folders found for {dataset_name}"}

            # Validate transforms
            try:
                training_transforms = get_transforms('training')
                validation_transforms = get_transforms('validation')
            except Exception as e:
                logger.error(f"Failed to load transforms: {e}")
                return {"status": StepStatus.ERROR.value, "error": f"Failed to load transforms: {str(e)}"}
            
            min_images_per_player = training_config.min_images_per_player

            # Create datasets using the enhanced LacrossePlayerDataset
            # Pass single string for single mode, list for multi mode
            training_dataset = LacrossePlayerDataset(
                image_dir=train_folders if dataset_mode == "single" else train_folders,
                storage_client=self.storage_client,
                transform=training_transforms,
                min_images_per_player=min_images_per_player
            )
            validation_dataset = LacrossePlayerDataset(
                image_dir=val_folders[0] if dataset_mode == "single" else val_folders,
                storage_client=self.storage_client,
                transform=validation_transforms,
                min_images_per_player=min_images_per_player
            )

            context.update({
                'training_dataset': training_dataset,
                'validation_dataset': validation_dataset,
                'dataset_mode': dataset_mode,
                'status': StepStatus.COMPLETED.value
            })
        
            # Log success with mode-specific information
            if dataset_mode == "single":
                logger.info(f"✅ Single dataset created successfully:")
                logger.info(f"   📁 Train folder: {train_folders[0]}")
                logger.info(f"   📁 Validation folder: {val_folders[0]}")
            else:
                logger.info(f"✅ Multi-dataset created successfully:")
                logger.info(f"   📁 Train folders: {len(train_folders)} datasets")
                logger.info(f"   📁 Validation folders: {len(val_folders)} datasets")
                for i, (train_f, val_f) in enumerate(zip(train_folders, val_folders)):
                    logger.info(f"      Dataset {i+1}: {train_f} | {val_f}")
            
            logger.info(f"   📏 Min images per player: {min_images_per_player}")
            logger.info(f"   🎯 Mode: {dataset_mode}-dataset with {'same-dataset negative mining' if dataset_mode == 'multi' else 'standard negative mining'}")
            
            return context
            
        except Exception as e:
            # Catch any unexpected errors
            logger.error(f"Unexpected error in dataset creation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"status": StepStatus.ERROR.value, "error": f"Unexpected error: {str(e)}"}
        

    def _train_model(self, context: dict, stop_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Train the neural network model using the Training class.
        
        Args:
            context: Pipeline context containing dataset and dataset_path
            
        Returns:
            Updated context with training results
        """
        try:
            
            # Check for required inputs
            training_dataset = context.get('training_dataset')
            val_dataset = context.get('validation_dataset')
            # Support legacy or mocked create_dataset that returns a tuple (train, val)
            if training_dataset is None and 'create_dataset_result' in context:
                res = context.get('create_dataset_result')
                if isinstance(res, (list, tuple)) and len(res) >= 2:
                    training_dataset, val_dataset = res[0], res[1]
            dataset_guid = context.get('dataset_name')
            resume_from_checkpoint = context.get('resume_from_checkpoint', False)
            custom_name = context.get('custom_name', 'run')

            logger.info(f"Starting model training with dataset from: {dataset_guid}")
            if resume_from_checkpoint:
                logger.info("Checkpoint resumption enabled - will check for existing checkpoints")

            # Initialize Training class (import at runtime so tests can patch training_loop.Training)
            try:
                from training_loop import Training as _RuntimeTraining
                training = _RuntimeTraining(
                    **self.training_kwargs
                )
            except Exception as e:
                logger.error(f"Failed to initialize Training class: {e}")
                return {"status": StepStatus.ERROR.value, "error": f"Failed to initialize Training class: {str(e)}"}

            # Execute complete training pipeline with checkpoint support
            logger.info("Executing training pipeline...")

            trained_model = training.train_and_save(
                model_class=self.model_class,
                dataset=training_dataset,
                model_name=self.collection_name,
                val_dataset=val_dataset,
                resume_from_checkpoint=resume_from_checkpoint,
                stop_callback=stop_callback
            )

            training_info = training.get_training_info()
            
            logger.info("✅ Model training completed successfully")
            logger.info(f"   Training device: {training_info['device']}")
            logger.info(f"   Model type: {type(trained_model)}")
            
            # Update context with training results
            context.update({
                'trained_model': trained_model,
                'training_info': training_info,
                'device': training_info['device'],
                'status': StepStatus.COMPLETED.value
            })
            
            return context
            
        except InterruptedError as e:
            logger.info(f"Model training cancelled by external request: {e}")
            # Re-raise InterruptedError to propagate cancellation up the chain
            raise
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"status": StepStatus.ERROR.value, "error": error_msg}


    def _evaluate_model(self, context: dict, stop_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation using multiple methodologies.
        
        Args:
            context: Pipeline context containing trained model and dataset
            
        Returns:
            Updated context with evaluation results
        """

        try:
            validation_dataset = context.get('validation_dataset')
            dataset_guid = context.get('dataset_name')
            trained_model = context.get('trained_model')
            device = context.get('device')

            
            # Check for required inputs
            if validation_dataset is None:
                error_msg = "Validation dataset must be available in context (run prepare_validation_dataset step first)"
                logger.error(error_msg)
                return {"status": StepStatus.ERROR.value, "error": error_msg}

            if trained_model is None:
                error_msg = "Trained model must be available in context (run train_model step first)"
                logger.error(error_msg)
                return {"status": StepStatus.ERROR.value, "error": error_msg}
            
            
            logger.info("Starting comprehensive model evaluation...")
            logger.info(f"Evaluation device: {device}")
            
            # Initialize evaluator
            try:
                # Prefer the module-level alias `Evaluator` so tests can patch it.
                evaluator_cls = Evaluator if 'Evaluator' in globals() else ModelEvaluator
                evaluator = evaluator_cls(
                    model=trained_model,
                    device=device,
                    stop_callback=stop_callback
                )
                logger.info("Model evaluator initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize model evaluator: {str(e)}"
                logger.error(error_msg)
                return {"status": StepStatus.ERROR.value, "error": error_msg}
            
            # Run evaluation. Some callers/tests mock an `evaluate` method on the
            # evaluator (legacy API); prefer that when available so mocks work.
            logger.info("Running evaluation suite...")
            if hasattr(evaluator, 'evaluate'):
                evaluation_results = evaluator.evaluate(validation_dataset, **self.training_kwargs)
            else:
                evaluation_results = evaluator.evaluate_comprehensive(validation_dataset, **self.training_kwargs)

            # Generate human-readable report if available and results contain
            # metrics; otherwise build a simple summary string. Some tests
            # mock an evaluator and may return empty metric dicts, so avoid
            # calling generate_evaluation_report in that case to prevent
            # KeyError inside the report builder.
            cls_metrics = evaluation_results.get('classification_metrics', {}) if isinstance(evaluation_results, dict) else {}
            rank_metrics = evaluation_results.get('ranking_metrics', {}) if isinstance(evaluation_results, dict) else {}
            if hasattr(evaluator, 'generate_evaluation_report') and (cls_metrics or rank_metrics):
                evaluation_report = evaluator.generate_evaluation_report(evaluation_results)
            else:
                evaluation_report = f"Evaluation results: {evaluation_results}"
            
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
                'status': StepStatus.COMPLETED.value
            })
            
            # Print report to console if verbose
            if self.verbose:
                print("\n" + evaluation_report)
            
            return context
            
        except InterruptedError as e:
            logger.info(f"Model evaluation cancelled by external request: {e}")
            # Re-raise InterruptedError to propagate cancellation up the chain
            raise
            
        except Exception as e:
            error_msg = f"Model evaluation failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"status": StepStatus.ERROR.value, "error": error_msg}