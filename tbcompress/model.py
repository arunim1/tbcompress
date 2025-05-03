# /// script
# dependencies = [
#     "torch",
# ]
# ///

import torch
import torch.nn as nn


class TablebaseModel(nn.Module):
    """NNUE-inspired neural network for lightweight tablebase classification"""

    def __init__(self, input_size=65, hidden_size=256, output_size=32, num_classes=3):
        """
        Initialize a tablebase model with configurable architecture.

        Args:
            input_size: Number of input features (default: 65 for 64 squares + side to move)
            hidden_size: Size of the feature transformer layer (default: 256)
            output_size: Size of the final hidden layer (default: 32)
            num_classes: Number of output classes (default: 3 for loss/draw/win)
        """
        super(TablebaseModel, self).__init__()

        # Feature transformer (first layer)
        self.feature_transformer = nn.Linear(input_size, hidden_size)

        # Use CELU activation for better properties than ReLU
        self.activation = nn.CELU(alpha=0.5)

        # Compact hidden layer
        self.hidden = nn.Linear(hidden_size, output_size)

        # Batch normalization for training stability and faster convergence
        self.batch_norm = nn.BatchNorm1d(output_size)

        # Output layer
        self.output = nn.Linear(output_size, num_classes)

    def forward(self, x):
        # Feature transformation
        x = self.activation(self.feature_transformer(x))

        # Hidden layer with batch normalization
        x = self.batch_norm(self.activation(self.hidden(x)))

        # Output layer
        return self.output(x)

    def get_num_parameters(self):
        """Returns the total number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())


def export_model(model, model_path):
    """
    Export the model to lightweight formats for deployment
    
    Args:
        model: The trained model to export
        model_path: Path where model will be saved (without extension)
    """
    # Save the PyTorch model (this is the primary format we need)
    print(f"Saving PyTorch model to {model_path}.pth")
    torch.save(model.state_dict(), f"{model_path}.pth")
    
    # Optionally export to TorchScript format
    try:
        # Convert to TorchScript for deployment
        model.eval()  # Set to evaluation mode
        example_input = torch.randn(1, 65)
        scripted_model = torch.jit.script(model)
        scripted_model.save(f"{model_path}.pt")
        print(f"Saved TorchScript model to {model_path}.pt")
    except Exception as e:
        print(f"Note: TorchScript export skipped: {e}")

    # Optionally export to ONNX format if available
    # We'll skip this if it causes errors since it's not essential
    try:
        import onnx  # Check if onnx is installed
        model.eval()
        example_input = torch.randn(1, 65)
        torch.onnx.export(
            model,
            example_input,
            f"{model_path}.onnx",
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"Saved ONNX model to {model_path}.onnx")
    except ImportError:
        # ONNX not installed, skip silently
        pass
    except Exception as e:
        # Other ONNX errors, print message but continue
        print(f"Note: ONNX export skipped: {e}")
