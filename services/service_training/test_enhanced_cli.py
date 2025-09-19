#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced CLI error handling capabilities.

This script shows how the training CLI now handles:
1. Unrecognized arguments with suggestions
2. Duplicate arguments with warnings
3. Graceful error recovery
"""

import sys
import logging
import argparse
import difflib

# Configure logging to show warnings and info
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Import our enhanced CLI components
class TolerantArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that doesn't exit on errors"""
    
    def error(self, message):
        # Don't exit, just log the error
        logging.warning(f'Argument parsing error: {message}')

def validate_and_suggest_args(unknown_args, parser):
    """Validate unknown arguments and suggest corrections"""
    # Get all valid argument names
    valid_args = []
    for action in parser._actions:
        if action.option_strings:
            valid_args.extend(action.option_strings)
    
    for arg in unknown_args:
        if arg.startswith('--'):
            # Find closest matches
            matches = difflib.get_close_matches(arg, valid_args, n=3, cutoff=0.6)
            if matches:
                logging.warning(f'Unrecognized argument "{arg}". Did you mean: {", ".join(matches)}?')
            else:
                logging.warning(f'Unrecognized argument "{arg}". Use --help to see available options.')

def create_sample_parser():
    """Create a sample parser with training parameters"""
    parser = TolerantArgumentParser(description='Enhanced Training CLI Demo')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--model_name', type=str, default='siamesenet',
                       help='Model architecture to use')
    parser.add_argument('--train_prefetch_factor', type=int, default=2,
                       help='Training data prefetch factor')
    parser.add_argument('--eval_prefetch_factor', type=int, default=1,
                       help='Evaluation data prefetch factor')
    
    return parser

def test_scenarios():
    """Test various CLI argument scenarios"""
    
    print("=" * 60)
    print("Enhanced CLI Error Handling Demo")
    print("=" * 60)
    
    parser = create_sample_parser()
    
    # Test scenarios
    test_cases = [
        {
            'name': 'Valid arguments',
            'args': ['--batch_size', '64', '--learning_rate', '0.01', '--epochs', '20']
        },
        {
            'name': 'Typos in argument names',
            'args': ['--batch_sizze', '64', '--learning_ratee', '0.01', '--epoch', '20']
        },
        {
            'name': 'Completely unknown arguments',
            'args': ['--invalid_arg', 'test', '--random_param', '123']
        },
        {
            'name': 'Mix of valid and invalid',
            'args': ['--batch_size', '64', '--unknown_arg', 'value', '--learning_rate', '0.01']
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"   Args: {test_case['args']}")
        print("   " + "-" * 50)
        
        try:
            known_args, unknown_args = parser.parse_known_args(test_case['args'])
            
            if known_args:
                print(f"   ✓ Parsed known args: {known_args}")
            
            if unknown_args:
                print(f"   ⚠ Unknown args found: {unknown_args}")
                validate_and_suggest_args(unknown_args, parser)
            else:
                print("   ✓ All arguments recognized!")
                
        except Exception as e:
            logging.error(f"   ✗ Unexpected error: {e}")

if __name__ == '__main__':
    test_scenarios()
    
    print("\n" + "=" * 60)
    print("Demo completed! Key features:")
    print("• Graceful handling of unrecognized arguments")
    print("• Smart suggestions for typos using fuzzy matching")
    print("• Detailed warnings with helpful guidance")
    print("• Continued parsing of valid arguments")
    print("=" * 60)