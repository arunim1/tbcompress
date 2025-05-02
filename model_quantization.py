# /// script
# dependencies = [
#     "torch",
# ]
# ///

"""
Quantization utilities for tablebase classifier models.
This file contains functions to quantize and optimize models for deployment.
"""

import torch
import torch.nn as nn
import os
import sys


def quantize_model(model, dtype=torch.qint8):
    """
    Convert model to quantized version for inference.

    Args:
        model: The PyTorch model to quantize
        dtype: Quantization data type (default: torch.qint8)

    Returns:
        Quantized model if successful, original model if not
    """
    try:
        # Make sure model is in eval mode
        model.eval()

        # Check if quantization is supported
        if not hasattr(torch, "quantization"):
            print("Warning: PyTorch quantization not available")
            return model

        # Quantize model to int8 for smaller size and faster inference
        print(f"Quantizing model to {dtype}...")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=dtype
        )

        # Calculate size reduction
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        quantized_size = sum(
            p.numel() * p.element_size() for p in quantized_model.parameters()
        )
        reduction = (1 - quantized_size / original_size) * 100

        print(
            f"Model quantized: {original_size/1024:.1f}KB → {quantized_size/1024:.1f}KB ({reduction:.1f}% reduction)"
        )
        return quantized_model

    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Returning original model")
        return model


def export_quantized_model(
    model, model_path="lightweight_tablebase_classifier_quantized"
):
    """
    Export a quantized version of the model if possible.

    Args:
        model: The PyTorch model to quantize and export
        model_path: Path to save the quantized model
    """
    try:
        # Try to quantize the model
        quantized_model = quantize_model(model)

        # Save the quantized model
        torch.save(quantized_model.state_dict(), f"{model_path}.pth")
        print(f"Saved quantized model to {model_path}.pth")

        return True
    except Exception as e:
        print(f"Failed to export quantized model: {e}")
        return False


if __name__ == "__main__":
    # This allows the script to be run directly to quantize an existing model
    if len(sys.argv) < 2:
        print("Usage: python model_quantization.py <model_path> [output_path]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path + "_quantized"

    if not os.path.exists(input_path):
        print(f"Error: Model file {input_path} not found")
        sys.exit(1)

    print(f"Loading model from {input_path}")

    try:
        # We need to import the model class
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from tablebase_classifier import OptimizedTablebaseClassifier

        # Create a model instance
        model = OptimizedTablebaseClassifier()

        # Load the state dict
        model.load_state_dict(torch.load(input_path))

        # Quantize and export
        export_quantized_model(model, output_path)

        print(f"Quantization complete. Model saved to {output_path}.pth")
    except Exception as e:
        print(f"Error during quantization: {e}")
        sys.exit(1)
