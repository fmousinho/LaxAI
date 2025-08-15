#!/usr/bin/env python3
"""Test script to verify parameter registry functionality"""

from src.config.parameter_registry import parameter_registry

def test_parameter_registry():
    """Test that parameter registry returns different sets correctly"""
    
    # Test getting all parameters
    all_params = parameter_registry.parameters
    print(f"Total parameters: {len(all_params)}")
    
    # Test getting training parameters only
    training_params = parameter_registry.get_training_parameters()
    print(f"Training parameters: {len(training_params)}")
    print("Training parameter names:", list(training_params.keys())[:5], "...")  # Show first 5
    
    # Test getting model parameters only
    model_params = parameter_registry.get_model_parameters()
    print(f"Model parameters: {len(model_params)}")
    print("Model parameter names:", list(model_params.keys()))
    
    # Verify they don't overlap
    training_names = set(training_params.keys())
    model_names = set(model_params.keys())
    overlap = training_names & model_names
    print(f"Parameter overlap: {overlap}")
    
    # Verify total equals sum of parts
    total_separate = len(training_params) + len(model_params)
    print(f"Sum of separate collections: {total_separate}")
    print(f"Total in registry: {len(all_params)}")
    print(f"Counts match: {total_separate == len(all_params)}")
    
    # Test CLI parser generation for training only
    try:
        training_parser = parameter_registry.generate_cli_parser_for_training()
        print(f"Training CLI parser created successfully")
    except Exception as e:
        print(f"Error creating training CLI parser: {e}")
    
    # Test CLI parser generation for model only
    try:
        model_parser = parameter_registry.generate_cli_parser_for_model()
        print(f"Model CLI parser created successfully")
    except Exception as e:
        print(f"Error creating model CLI parser: {e}")
    
    # Test Pydantic field generation for training only
    try:
        training_fields = parameter_registry.generate_pydantic_fields_for_training()
        print(f"Training Pydantic fields: {len(training_fields)} fields created")
    except Exception as e:
        print(f"Error creating training Pydantic fields: {e}")
    
    # Test Pydantic field generation for model only
    try:
        model_fields = parameter_registry.generate_pydantic_fields_for_model()
        print(f"Model Pydantic fields: {len(model_fields)} fields created")
    except Exception as e:
        print(f"Error creating model Pydantic fields: {e}")

if __name__ == "__main__":
    test_parameter_registry()
