"""
Train a single tablebase model for a specific .rtbw file
"""

import os
import sys
import argparse
import traceback
from tbcompress.streaming_dataset import StreamingTablebaseDataset
from tbcompress.model import TablebaseModel
from tbcompress.train import train_and_save_model


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Train a model for a specific tablebase file"
        )
        parser.add_argument("--rtbw-file", required=True, help="Path to .rtbw file")
        parser.add_argument(
            "--tablebase-dir",
            required=True,
            help="Directory containing tablebase files",
        )
        parser.add_argument(
            "--output-dir", default="models", help="Directory to save trained models"
        )
        parser.add_argument(
            "--hidden-size", type=int, default=128, help="Hidden layer size"
        )
        parser.add_argument(
            "--output-size", type=int, default=32, help="Output layer size"
        )
        parser.add_argument(
            "--batch-size", type=int, default=4096, help="Batch size for training"
        )
        parser.add_argument(
            "--epochs", type=int, default=2, help="Maximum number of training epochs"
        )
        parser.add_argument(
            "--learning-rate", type=float, default=0.001, help="Learning rate"
        )
        args = parser.parse_args()

        # Validate inputs
        if not os.path.exists(args.rtbw_file):
            raise FileNotFoundError(f"RTBW file not found: {args.rtbw_file}")

        if not os.path.isfile(args.rtbw_file) or not args.rtbw_file.endswith(".rtbw"):
            raise ValueError(f"Invalid RTBW file: {args.rtbw_file}")

        if not os.path.exists(args.tablebase_dir):
            raise FileNotFoundError(
                f"Tablebase directory not found: {args.tablebase_dir}"
            )

        if not os.path.isdir(args.tablebase_dir):
            raise NotADirectoryError(f"Not a directory: {args.tablebase_dir}")

        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Get model name from file name (without extension)
        model_name = os.path.basename(args.rtbw_file).split(".")[0]

        print(f"Starting training for {model_name}...")

        # Create streaming dataset from the specific .rtbw file
        try:
            # Use streaming dataset to avoid loading all positions into memory
            # The cache_size parameter controls memory usage - increase/decrease as needed
            cache_size = 50000  # Smaller cache = less memory but more frequent refills
            max_training_positions = (
                5e8  # Set an unreasonable limit for the total positions to train on
            )

            print(
                f"Creating streaming dataset for {args.rtbw_file} with cache size {cache_size}"
            )
            dataset = StreamingTablebaseDataset(
                rtbw_file=args.rtbw_file,
                tablebase_dir=args.tablebase_dir,
                cache_size=cache_size,
                max_positions=max_training_positions,
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to create streaming dataset from {args.rtbw_file}: {str(e)}"
            )

        # The streaming dataset will generate positions as needed, so we don't need to check size

        # Create model
        model = TablebaseModel(
            input_size=769,  # 64 squares + side to move
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            num_classes=3,  # Loss, Draw, Win
        )

        try:
            # Train and save model
            result = train_and_save_model(
                model=model,
                train_dataset=dataset,
                output_path=args.output_dir,
                model_name=model_name,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )

            # Verify the model was saved
            model_path = os.path.join(args.output_dir, f"{model_name}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Expected model file not found: {model_path}")

            # Print summary
            print("\nTraining Summary:")
            print(f"Model: {result['model_name']}")
            print(f"Parameters: {result['parameters']:,}")
            print(f"Final Accuracy: {result['final_accuracy']:.4f}")
            print(f"Training Time: {result['training_time']:.2f} seconds")
            print(f"Model saved to: {result['model_path']}")
        finally:
            # Ensure the dataset's producer thread is properly shut down
            if hasattr(dataset, "shutdown"):
                print("Shutting down dataset producer thread...")
                dataset.shutdown()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
