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
