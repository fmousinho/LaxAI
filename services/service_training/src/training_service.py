"""
Training service utilities for service_training module.

This module provides utility functions for handling training requests
and converting them to the appropriate format for the training workflow.
"""
from typing import Any, Dict


def _convert_request_to_kwargs(request: Any) -> Dict[str, Any]:
    """
    Convert a training request object to kwargs for the train function.

    Args:
        request: Request object with training parameters.

    Returns:
        Dictionary of kwargs suitable for the train function.
    """
    kwargs = {}

    # Extract basic parameters
    if hasattr(request, 'tenant_id'):
        kwargs['tenant_id'] = request.tenant_id
    if hasattr(request, 'verbose'):
        kwargs['verbose'] = request.verbose
    if hasattr(request, 'custom_name'):
        kwargs['custom_name'] = request.custom_name
    if hasattr(request, 'resume_from_checkpoint'):
        kwargs['resume_from_checkpoint'] = request.resume_from_checkpoint
    if hasattr(request, 'wandb_tags'):
        kwargs['wandb_tags'] = request.wandb_tags
    if hasattr(request, 'n_datasets_to_use'):
        kwargs['n_datasets_to_use'] = request.n_datasets_to_use

    # Handle training and model parameters
    if hasattr(request, 'training_params') and request.training_params:
        kwargs['training_kwargs'] = request.training_params
    if hasattr(request, 'model_params') and request.model_params:
        kwargs['model_kwargs'] = request.model_params

    return kwargs
