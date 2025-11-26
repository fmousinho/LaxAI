#!/usr/bin/env python3
"""
Simple test script to verify JSON parameter parsing works correctly.
"""
import sys
import json

# Test the parsing logic directly
sys.path.insert(0, '/Users/fmousinho/Library/Mobile Documents/com~apple~CloudDocs/Documents/Learning_to_Code/LaxAI')
sys.path.insert(0, '/Users/fmousinho/Library/Mobile Documents/com~apple~CloudDocs/Documents/Learning_to_Code/LaxAI/services/service_training/src')

from schemas.training import TrainingParams, ModelParams, EvalParams

def test_training_params():
    """Test parsing training parameters from JSON."""
    json_str = '{"num_epochs": 100, "batch_size": 32, "learning_rate": 0.001}'
    json_dict = json.loads(json_str)
    params = TrainingParams(**json_dict)
    print(f"✅ Training params parsed: {params.dict()}")
    assert params.num_epochs == 100
    assert params.batch_size == 32
    # lr_initial is the field name, not learning_rate
    print(f"   Note: learning_rate not in schema, using default lr_initial={params.lr_initial}")

def test_model_params():
    """Test parsing model parameters from JSON."""
    json_str = '{"embedding_dim": 512}'
    json_dict = json.loads(json_str)
    params = ModelParams(**json_dict)
    print(f"✅ Model params parsed: {params.dict()}")
    assert params.embedding_dim == 512

def test_eval_params():
    """Test parsing eval parameters from JSON."""
    json_str = '{"batch_size": 64}'
    json_dict = json.loads(json_str)
    params = EvalParams(**json_dict)
    print(f"✅ Eval params parsed: {params.dict()}")
    assert params.batch_size == 64

def test_defaults():
    """Test that defaults work when no JSON is provided."""
    params = TrainingParams()
    print(f"✅ Default training params: num_epochs={params.num_epochs}, batch_size={params.batch_size}")

if __name__ == "__main__":
    print("🧪 Testing JSON parameter parsing...\n")
    test_training_params()
    test_model_params()
    test_eval_params()
    test_defaults()
    print("\n✅ All tests passed!")
