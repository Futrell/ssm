#!/usr/bin/env python3
"""
Test script to verify that grammatical filtering is working correctly.
"""

import subprocess
import sys
import os

def test_filtering():
    # Test with filtering enabled (default)
    print("Testing with grammatical filtering enabled...")
    result = subprocess.run([
        sys.executable, 'eval_model.py', 'sl2',
        'data/converted_mlregtest/TSL.2.1.6/TSL.2.1.6LearningData.txt',
        'data/converted_mlregtest/TSL.2.1.6/TSL.2.1.6TestingPairs.txt',
        '--batch_size', '2',
        '--num_epochs', '1',
        '--lr', '0.01',
        '--test_data_paired'
    ], capture_output=True, text=True, cwd='/Users/huteng/Desktop/ssm')

    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    # Test without filtering
    print("\nTesting with grammatical filtering disabled...")
    result2 = subprocess.run([
        sys.executable, 'eval_model.py', 'sl2',
        'data/converted_mlregtest/TSL.2.1.6/TSL.2.1.6LearningData.txt',
        'data/converted_mlregtest/TSL.2.1.6/TSL.2.1.6TestingPairs.txt',
        '--batch_size', '2',
        '--num_epochs', '1',
        '--lr', '0.01',
        '--test_data_paired',
        '--no-filter_training_grammatical'
    ], capture_output=True, text=True, cwd='/Users/huteng/Desktop/ssm')

    print("STDOUT:")
    print(result2.stdout)
    if result2.stderr:
        print("STDERR:")
        print(result2.stderr)

if __name__ == "__main__":
    test_filtering()
