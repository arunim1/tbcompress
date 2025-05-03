# /// script
# dependencies = [
#     "torch",
#     "numpy",
#     "python-chess",
#     "tqdm",
# ]
# ///

"""
Parallel training script for tablebase compression models

This script trains a separate model for each .rtbw file in the tablebase directory,
running up to 8 training jobs in parallel.
"""

import os
import argparse
import glob
import concurrent.futures
import subprocess
import time
from datetime import datetime


def train_model_for_file(
    rtbw_file,
    tablebase_dir,
    output_dir,
    hidden_size,
    output_size,
    batch_size,
    epochs,
    learning_rate,
):
    """Run training for a single .rtbw file using the train_single_model.py script"""

    model_name = os.path.basename(rtbw_file).split(".")[0]
    output_path = os.path.join(output_dir, f"{model_name}.pth")

    # Skip if model already exists
    if os.path.exists(output_path):
        print(f"Model for {model_name} already exists at {output_path}, skipping")
        return {
            "model_name": model_name,
            "status": "skipped",
            "file": rtbw_file,
        }

    # Command to run the training script
    cmd = [
        "uv",
        "run",
        "train_single_model.py",
        "--rtbw-file",
        rtbw_file,
        "--tablebase-dir",
        tablebase_dir,
        "--output-dir",
        output_dir,
        "--hidden-size",
        str(hidden_size),
        "--output-size",
        str(output_size),
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(learning_rate),
    ]

    try:
        # Run the training process
        print(f"Starting training for {model_name}...")
        start_time = time.time()

        # Execute the process and capture output
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        training_time = time.time() - start_time

        # Check if the model file was actually created
        if os.path.exists(output_path):
            return {
                "model_name": model_name,
                "status": "completed",
                "file": rtbw_file,
                "training_time": training_time,
                "output": process.stdout,
            }
        else:
            # Process completed but model file wasn't created
            print(f"Warning: Process for {model_name} completed but model file was not created.")
            return {
                "model_name": model_name,
                "status": "failed",
                "file": rtbw_file,
                "error": "Model file was not created",
                "output": process.stdout,
            }

    except subprocess.CalledProcessError as e:
        error_msg = f"Error training model for {model_name}: {e}"
        print(error_msg)
        print(f"Error output: {e.stderr}")
        return {
            "model_name": model_name,
            "status": "failed",
            "file": rtbw_file,
            "error": str(e),
            "output": e.stdout,
            "error_output": e.stderr,
        }
    except Exception as e:
        # Catch any other exceptions that might occur
        error_msg = f"Unexpected error training model for {model_name}: {str(e)}"
        print(error_msg)
        return {
            "model_name": model_name,
            "status": "failed",
            "file": rtbw_file,
            "error": str(e),
            "output": "",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Train models for multiple tablebase files in parallel"
    )
    parser.add_argument(
        "--tablebase-dir", required=True, help="Directory containing tablebase files"
    )
    parser.add_argument(
        "--output-dir", default="models", help="Directory to save trained models"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=128, help="Hidden layer size"
    )
    parser.add_argument("--output-size", type=int, default=32, help="Output layer size")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel training jobs",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Get all .rtbw files
    rtbw_files = glob.glob(os.path.join(args.tablebase_dir, "*.rtbw"))
    if not rtbw_files:
        print(f"Error: No .rtbw files found in {args.tablebase_dir}")
        return

    print(f"Found {len(rtbw_files)} .rtbw files to process")

    # Create a log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"training_log_{timestamp}.txt")

    with open(log_file, "w") as f:
        f.write(f"Training started at: {datetime.now()}\n")
        f.write(f"Tablebase directory: {args.tablebase_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Number of .rtbw files: {len(rtbw_files)}\n")
        f.write(
            f"Model parameters: hidden_size={args.hidden_size}, output_size={args.output_size}\n"
        )
        f.write(
            f"Training parameters: batch_size={args.batch_size}, epochs={args.epochs}, learning_rate={args.learning_rate}\n"
        )
        f.write(f"Maximum parallel workers: {args.max_workers}\n\n")

    # Track results
    results = []

    # Train models in parallel
    print(f"Starting parallel training with {args.max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        # Submit all training jobs
        future_to_file = {
            executor.submit(
                train_model_for_file,
                rtbw_file,
                args.tablebase_dir,
                args.output_dir,
                args.hidden_size,
                args.output_size,
                args.batch_size,
                args.epochs,
                args.learning_rate,
            ): rtbw_file
            for rtbw_file in rtbw_files
        }

        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
            rtbw_file = future_to_file[future]
            model_name = os.path.basename(rtbw_file).split(".")[0]

            try:
                result = future.result()
                results.append(result)

                # Log completion
                with open(log_file, "a") as f:
                    f.write(
                        f"[{i+1}/{len(rtbw_files)}] {model_name}: {result['status']}\n"
                    )
                    if result["status"] == "completed":
                        f.write(
                            f"  Training time: {result.get('training_time', 'N/A'):.2f} seconds\n"
                        )
                    elif result["status"] == "failed":
                        f.write(f"  Error: {result.get('error', 'Unknown error')}\n")

                # Print progress
                print(f"[{i+1}/{len(rtbw_files)}] {model_name}: {result['status']}")

            except Exception as e:
                print(f"Error processing {model_name}: {e}")
                with open(log_file, "a") as f:
                    f.write(
                        f"[{i+1}/{len(rtbw_files)}] {model_name}: error - {str(e)}\n"
                    )

    # Summarize results
    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print("\nTraining Summary:")
    print(f"Total files: {len(rtbw_files)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped: {len(skipped)}")

    # Write summary to log
    with open(log_file, "a") as f:
        f.write("\nTraining Summary:\n")
        f.write(f"Total files: {len(rtbw_files)}\n")
        f.write(f"Completed: {len(completed)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Skipped: {len(skipped)}\n")
        f.write(f"\nTraining completed at: {datetime.now()}\n")

    print(f"Log file saved to: {log_file}")


if __name__ == "__main__":
    main()
