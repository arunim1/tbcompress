"""
Benchmark script for tablebase training performance
"""

import os
import sys
import time
import cProfile
import argparse
from tbcompress.streaming_dataset import StreamingTablebaseDataset
from tbcompress.model import TablebaseModel
from tbcompress.train import train_with_streaming_dataset


def benchmark_training():
    """Run a standardized training benchmark"""
    # Set fixed parameters for the benchmark
    rtbw_file = os.path.join(os.getcwd(), "Syzygy345_WDL", "KBvK.rtbw")
    tablebase_dir = os.path.join(os.getcwd(), "Syzygy345_WDL")
    batch_size = 1024
    max_steps = 500  # Aim for ~500 steps to complete in ~2 minutes

    # Create dataset with a smaller cache to ensure it runs through the data pipeline
    cache_size = 10000

    print(f"Creating dataset for {rtbw_file} with cache_size={cache_size}")

    # Create dataset
    try:
        dataset = StreamingTablebaseDataset(
            rtbw_file=rtbw_file,
            tablebase_dir=tablebase_dir,
            cache_size=cache_size,
            max_positions=None,
        )

        # Create model with minimal size for quick training
        model = TablebaseModel(
            input_size=769,  # 64 squares + side to move
            hidden_size=64,  # Smaller than usual to speed up training
            output_size=16,  # Smaller than usual to speed up training
            num_classes=3,  # Loss, Draw, Win
        )

        # Define training parameters
        learning_rate = 0.001

        # Time the training
        start_time = time.time()

        # Train for exactly max_steps steps
        result = train_with_streaming_dataset(
            model=model,
            train_dataset=dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_training_steps=max_steps,  # Cap at exactly this many steps
        )

        # Calculate metrics
        end_time = time.time()
        training_time = end_time - start_time
        steps_per_second = max_steps / training_time
        positions_per_second = max_steps * batch_size / training_time

        # Print results
        print("\nBenchmark Results:")
        print(f"Steps completed: {max_steps}")
        print(f"Batch size: {batch_size}")
        print(f"Total positions: {max_steps * batch_size:,}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Steps per second: {steps_per_second:.2f}")
        print(f"Positions per second: {positions_per_second:.2f}")

        return training_time, steps_per_second, positions_per_second

    finally:
        # Ensure the dataset's producer thread is properly shut down
        if "dataset" in locals() and hasattr(dataset, "shutdown"):
            print("Shutting down dataset producer thread...")
            dataset.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Run training benchmark")
    parser.add_argument("--profile", action="store_true", help="Run with profiler")
    args = parser.parse_args()

    if args.profile:
        # Run with profiler
        profile_output = "benchmark_profile.prof"
        print(f"Running with profiler. Output will be saved to {profile_output}")

        cProfile.run("benchmark_training()", profile_output)
        print(f"Profile data saved to {profile_output}")

    else:
        # Run normal benchmark
        benchmark_training()


if __name__ == "__main__":
    main()
