#!/usr/bin/env python3
"""
Automated MLregtest Dataset Training Pipeline - Working Version
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

def get_available_datasets(base_dir="data/converted_mlregtest"):
    """Get list of available MLregtest datasets."""
    datasets = []
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return datasets

    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name

            # Check for required files
            learning_file = dataset_dir / f"{dataset_name}LearningData.txt"
            testing_file = dataset_dir / f"{dataset_name}TestingData.txt"
            pairs_file = dataset_dir / f"{dataset_name}TestingPairs.tsv"

            if learning_file.exists() and testing_file.exists():
                datasets.append(dataset_name)
            else:
                print(f"⚠️ Skipping {dataset_name}: missing required files")

    return sorted(datasets)

def train_single_dataset(dataset_name, model_class="SL2", epochs=100, batch_size=16, lr=0.001, device=None):
    """Train a single model on a single dataset using build_and_train."""
    try:
        from main import build_and_train
        import shutil

        print(f"🚀 Training {model_class} on {dataset_name}...")

        # Handle path mismatch: build_and_train expects data/language/ but we have data/converted_mlregtest/Language/
        source_dir = Path("data/converted_mlregtest") / dataset_name
        target_dir = Path("data") / dataset_name.lower()

        source_learning = source_dir / f"{dataset_name}LearningData.txt"
        target_learning = target_dir / f"{dataset_name}LearningData.txt"

        if not source_learning.exists():
            raise FileNotFoundError(f"Source learning data not found: {source_learning}")

        # Create target directory and copy file
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_learning, target_learning)
        print(f"   📁 Data staged: {source_learning} -> {target_learning}")

        start_time = time.time()

        # Train the model
        model = build_and_train(
            lang=dataset_name,
            model_class=model_class,
            batch=batch_size,
            epochs=epochs,
        )

        # Clean up temporary files
        if target_learning.exists():
            target_learning.unlink()
        if target_dir.exists() and not any(target_dir.iterdir()):
            target_dir.rmdir()

        training_time = time.time() - start_time
        print(f"✅ {model_class} on {dataset_name} completed in {training_time:.1f}s")

        return {
            "dataset": dataset_name,
            "model": model_class,
            "status": "success",
            "training_time": training_time
        }

    except Exception as e:
        # Clean up on error too
        try:
            target_dir = Path("data") / dataset_name.lower()
            target_learning = target_dir / f"{dataset_name}LearningData.txt"
            if target_learning.exists():
                target_learning.unlink()
            if target_dir.exists() and not any(target_dir.iterdir()):
                target_dir.rmdir()
        except:
            pass

        error_msg = f"Error training {model_class} on {dataset_name}: {str(e)}"
        print(f"❌ {error_msg}")

        return {
            "dataset": dataset_name,
            "model": model_class,
            "status": "failed",
            "error": error_msg
        }

def main():
    parser = argparse.ArgumentParser(description="MLregtest Batch Training")
    parser.add_argument("--datasets", type=str, default=None,
                       help="Pattern to filter datasets (e.g., 'SL.2')")
    parser.add_argument("--model", type=str, default="SL2",
                       help="Model to train (default: SL2)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs (default: 100)")
    parser.add_argument("--device", type=str, default=None,
                       choices=["cpu", "cuda", "mps"],
                       help="Force specific device (cpu/cuda/mps). Default: auto-select best available.")
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test on first 3 datasets")
    parser.add_argument("--single", type=str, default=None,
                       help="Train only a single dataset (e.g., 'SL.2.1.0')")

    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        # auto-detect like in your learner
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"🖥️  Using device: {device}")

    # Single dataset mode
    if args.single:
        print(f"🎯 Single dataset mode: {args.single}")
        result = train_single_dataset(
            dataset_name=args.single,
            model_class=args.model,
            epochs=args.epochs,
            device=args.device
        )
        print(f"\nResult: {result}")
        return result

    # Get datasets
    all_datasets = get_available_datasets()

    if args.datasets:
        datasets = [d for d in all_datasets if args.datasets in d]
        print(f"🔍 Filtered datasets (matching '{args.datasets}'): {len(datasets)} found")
    else:
        datasets = all_datasets
        print(f"📊 All datasets: {len(datasets)} found")

    if args.quick_test:
        datasets = datasets[:3]
        print(f"⚡ Quick test mode: using first {len(datasets)} datasets")

    if not datasets:
        print("❌ No datasets found!")
        return

    # Train models
    results = []
    failed_datasets = []
    successful_datasets = []

    print(f"\n{'='*60}")
    print(f"🚀 STARTING BATCH TRAINING")
    print(f"{'='*60}")
    print(f"📋 Datasets: {len(datasets)}")
    print(f"🤖 Model: {args.model}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"{'='*60}")

    for i, dataset in enumerate(datasets, 1):
        print(f"\n{'='*50}")
        print(f"Processing {i}/{len(datasets)}: {dataset}")
        print(f"{'='*50}")

        result = train_single_dataset(
            dataset_name=dataset,
            model_class=args.model,
            epochs=args.epochs,
            device=args.device
        )
        results.append(result)

        if result["status"] == "success":
            successful_datasets.append(dataset)
        else:
            failed_datasets.append(dataset)

    # Summary
    successful = len(successful_datasets)
    failed = len(failed_datasets)

    print(f"\n{'='*60}")
    print(f"📊 BATCH TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    if successful_datasets:
        print(f"\n✅ Successful datasets:")
        for dataset in successful_datasets:
            print(f"   • {dataset}")

    if failed_datasets:
        print(f"\n❌ Failed datasets:")
        for dataset in failed_datasets:
            print(f"   • {dataset}")

    # Save results
    results_file = Path("output") / f"batch_training_results_{args.model}_{len(datasets)}_datasets.json"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")

    return results

if __name__ == "__main__":
    main()

# Train SL2 on a single dataset
# python3 mlregtest_batch_working.py --single SL.2.1.0 --model SL2 --epochs 50

# Train SP2 on multiple datasets
# python3 mlregtest_batch_working.py --datasets "SP.2" --model 256 --epochs 500

# Train SYM model on all datasets
# python3 mlregtest_batch_working.py --model SYM --epochs 200

# Train 6-state PFA model
# python3 mlregtest_batch_working.py --single SL.2.1.0 --model 6 --epochs 100
# python3 mlregtest_batch_working.py --single SP.2.1.0 --model SP2 --epochs 3
