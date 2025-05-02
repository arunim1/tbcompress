# /// script
# dependencies = [
#     "torch",
#     "numpy",
#     "python-chess",
#     "tqdm",
# ]
# ///

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import random
import time
import chess
import chess.syzygy
import glob
from tqdm import tqdm
import pathlib

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Define an NNUE-inspired neural network for lightweight tablebase classification
class OptimizedTablebaseClassifier(nn.Module):
    def __init__(self, input_size=65, num_classes=3):
        super(OptimizedTablebaseClassifier, self).__init__()

        # Feature transformer (first layer) - inspired by NNUE architecture
        self.feature_transformer = nn.Linear(input_size, 256)

        # Use CELU activation for better properties than ReLU
        # Alpha=0.5 provides a good balance between ReLU and ELU
        self.activation = nn.CELU(alpha=0.5)

        # Single compact hidden layer
        self.hidden = nn.Linear(256, 32)

        # Batch normalization for better training stability and faster convergence
        self.batch_norm = nn.BatchNorm1d(32)

        # Output layer
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        # Feature transformation
        x = self.activation(self.feature_transformer(x))

        # Hidden layer with batch normalization
        x = self.batch_norm(self.activation(self.hidden(x)))

        # Output layer
        return self.output(x)


# Legacy classifier for backward compatibility
class TablebaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TablebaseClassifier, self).__init__()
        print(
            "Warning: Using legacy TablebaseClassifier. Consider using OptimizedTablebaseClassifier instead."
        )
        # Reduce hidden size to approximately 1/10th
        small_hidden = hidden_size // 10

        self.layers = nn.Sequential(
            nn.Linear(input_size, small_hidden),
            nn.ReLU(),
            nn.Linear(small_hidden, small_hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(small_hidden, small_hidden // 2),
            nn.ReLU(),
            nn.Linear(small_hidden // 2, small_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(small_hidden // 2, small_hidden // 4),
            nn.ReLU(),
            nn.Linear(small_hidden // 4, small_hidden // 4),
            nn.ReLU(),
            nn.Linear(small_hidden // 4, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


# Function to convert a chess position to feature vector - optimized for NNUE-inspired approach
def optimized_board_to_feature_vector(board):
    """Convert a chess.Board to a feature vector for the neural network using NNUE-inspired approach."""
    # Create a 64-element vector (one per square)
    # Empty square: 0
    # White pieces: 1 (pawn), 2 (knight), 3 (bishop), 4 (rook), 5 (queen), 6 (king)
    # Black pieces: -1 (pawn), -2 (knight), -3 (bishop), -4 (rook), -5 (queen), -6 (king)
    feature_vector = np.zeros(64, dtype=np.float32)

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }

    # Sparse representation - only populate non-empty squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = piece_values[piece.piece_type]
            if piece.color == chess.BLACK:
                value = -value
            feature_vector[square] = value

    # Side to move
    side_to_move = 1.0 if board.turn == chess.WHITE else -1.0

    # Combine features
    combined_features = np.append(feature_vector, side_to_move)
    return combined_features


# Legacy function for backward compatibility
def board_to_feature_vector(board):
    """Convert a chess.Board to a feature vector for the neural network."""
    return optimized_board_to_feature_vector(board)


# Real tablebase dataset
class SyzygyTablebaseDataset(Dataset):
    def __init__(self, tablebase_dir, max_positions=20000, cache_file=None):
        """
        Create a dataset from Syzygy tablebase files in the specified directory.

        Args:
            tablebase_dir: Directory containing .rtbw (WDL) files
            max_positions: Maximum number of positions to sample
            cache_file: If provided, try to load cached data from this file
        """
        self.tablebase_dir = tablebase_dir
        self.max_positions = max_positions

        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached tablebase data from {cache_file}")
            # Set weights_only=False to handle PyTorch 2.6 changes
            data = torch.load(cache_file, weights_only=False)
            self.features = data["features"]
            self.labels = data["labels"]
            print(f"Loaded {len(self.labels)} positions from cache")
        else:
            self.features, self.labels = self._load_tablebase_data()

            if cache_file:
                print(f"Saving tablebase data to cache: {cache_file}")
                torch.save(
                    {"features": self.features, "labels": self.labels}, cache_file
                )

    def _load_tablebase_data(self):
        """Load and parse the tablebase data."""
        print(f"Loading tablebase data from {self.tablebase_dir}")

        # Initialize the tablebase
        tablebase = chess.syzygy.open_tablebase(self.tablebase_dir)

        # Get all rtbw files
        rtbw_files = glob.glob(os.path.join(self.tablebase_dir, "*.rtbw"))
        print(f"Found {len(rtbw_files)} tablebase files")

        # We'll sample positions from the tablebases
        features = []
        labels = []

        # Sample positions from different material configurations
        positions_per_file = max(1, min(500, self.max_positions // len(rtbw_files)))

        for rtbw_file in tqdm(rtbw_files, desc="Processing tablebase files"):
            # Extract material configuration from filename (e.g., KRvKN.rtbw)
            material = os.path.basename(rtbw_file).split(".")[0]

            # Generate some positions with this material configuration
            for _ in range(positions_per_file):
                # Create a random position with this material configuration
                board = self._create_random_position_with_material(material)

                if board is not None:
                    try:
                        # Probe the tablebase for WDL value
                        wdl = tablebase.probe_wdl(board)

                        # Convert WDL to label (0=loss, 1=draw, 2=win)
                        # WDL values: -2, -1 (loss), 0 (draw), 1, 2 (win)
                        if wdl < 0:
                            label = 0  # loss
                        elif wdl > 0:
                            label = 2  # win
                        else:
                            label = 1  # draw

                        # Convert board to feature vector
                        feature_vector = board_to_feature_vector(board)

                        features.append(feature_vector)
                        labels.append(label)

                        if len(labels) >= self.max_positions:
                            break
                    except chess.syzygy.MissingTableError:
                        continue

            if len(labels) >= self.max_positions:
                break

        print(f"Loaded {len(labels)} positions from tablebases")
        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _create_random_position_with_material(self, material):
        """Create a random legal position with the specified material configuration."""
        # Parse material string (e.g., KRvKN -> King+Rook vs King+Knight)
        if "v" not in material:
            return None

        white_pieces, black_pieces = material.split("v")

        # Create empty board
        board = chess.Board(fen=None)
        board.clear_board()

        # Available squares
        available_squares = list(chess.SQUARES)
        random.shuffle(available_squares)

        # Place white king
        if "K" in white_pieces:
            white_king_square = available_squares.pop()
            board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))
            white_pieces = white_pieces.replace("K", "", 1)

        # Place black king
        if "K" in black_pieces:
            # Kings must be at least 2 squares apart
            valid_squares = [
                sq
                for sq in available_squares
                if chess.square_distance(sq, white_king_square) > 1
            ]
            if not valid_squares:
                return None
            black_king_square = random.choice(valid_squares)
            available_squares.remove(black_king_square)
            board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))
            black_pieces = black_pieces.replace("K", "", 1)

        # Map piece characters to piece types
        piece_map = {
            "Q": chess.QUEEN,
            "R": chess.ROOK,
            "B": chess.BISHOP,
            "N": chess.KNIGHT,
            "P": chess.PAWN,
        }

        # Place remaining white pieces
        for char in white_pieces:
            if char in piece_map and available_squares:
                square = available_squares.pop()
                # Pawns can't be on first or last rank
                if piece_map[char] == chess.PAWN:
                    pawn_ranks = [
                        chess.square_rank(sq)
                        for sq in available_squares
                        if chess.square_rank(sq) > 0 and chess.square_rank(sq) < 7
                    ]
                    if not pawn_ranks:
                        return None
                    valid_squares = [
                        sq
                        for sq in available_squares
                        if chess.square_rank(sq) > 0 and chess.square_rank(sq) < 7
                    ]
                    if not valid_squares:
                        return None
                    square = random.choice(valid_squares)
                    available_squares.remove(square)

                board.set_piece_at(square, chess.Piece(piece_map[char], chess.WHITE))

        # Place remaining black pieces
        for char in black_pieces:
            if char in piece_map and available_squares:
                square = available_squares.pop()
                # Pawns can't be on first or last rank
                if piece_map[char] == chess.PAWN:
                    valid_squares = [
                        sq
                        for sq in available_squares
                        if chess.square_rank(sq) > 0 and chess.square_rank(sq) < 7
                    ]
                    if not valid_squares:
                        return None
                    square = random.choice(valid_squares)
                    available_squares.remove(square)

                board.set_piece_at(square, chess.Piece(piece_map[char], chess.BLACK))

        # Random side to move
        board.turn = random.choice([chess.WHITE, chess.BLACK])

        # Check if position is legal
        if board.is_valid() and not board.is_check():
            return board
        return None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


# Optimized training function with mixed precision and learning rate scheduling
def train_optimized_classifier(
    model, train_loader, num_epochs=100, accuracy_threshold=0.95, learning_rate=0.001
):
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration")
        # Enable mixed precision training with CUDA
        use_amp = True
        scaler = torch.cuda.amp.GradScaler()
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     # Check if MPS (Metal Performance Shaders) is available for Mac Silicon GPU acceleration
    #     device = torch.device("mps")
    #     print("Using MPS (Metal Performance Shaders) for Mac Silicon GPU acceleration")
    #     use_amp = False  # MPS doesn't support AMP yet
    else:
        device = torch.device("cpu")
        print("Using CPU for training")
        use_amp = False

    model.to(device)

    # Use a more efficient optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Use a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    # Use cross entropy loss
    criterion = nn.CrossEntropyLoss()

    training_history = {"train_loss": [], "train_acc": []}
    best_train_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision where available
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        training_history["train_loss"].append(epoch_train_loss)
        training_history["train_acc"].append(epoch_train_acc)

        # Update learning rate based on loss
        scheduler.step(epoch_train_loss)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save the best model
        if epoch_train_acc > best_train_acc:
            best_train_acc = epoch_train_acc
            torch.save(model.state_dict(), "best_tablebase_model.pth")
            print(f"Model saved with training accuracy: {best_train_acc:.4f}")

        # Early stopping condition
        if epoch_train_acc >= accuracy_threshold:
            print(
                f"Reached {accuracy_threshold*100:.1f}% accuracy at epoch {epoch+1}. Early stopping."
            )
            break

    return training_history


# Legacy training function for backward compatibility
def train_tablebase_classifier(
    model, train_loader, criterion, optimizer, num_epochs=500, accuracy_threshold=0.95
):
    print(
        "Warning: Using legacy training function. Consider using train_optimized_classifier instead."
    )
    # if cuda, use cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check if MPS (Metal Performance Shaders) is available for Mac Silicon GPU acceleration
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for Mac Silicon GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU for training")

    model.to(device)

    training_history = {"train_loss": [], "train_acc": []}
    best_train_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        training_history["train_loss"].append(epoch_train_loss)
        training_history["train_acc"].append(epoch_train_acc)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")

        # Save the best model
        if epoch_train_acc > best_train_acc:
            best_train_acc = epoch_train_acc
            torch.save(model.state_dict(), "best_tablebase_model.pth")
            print(f"Model saved with training accuracy: {best_train_acc:.4f}")

        # Early stopping condition: stop at 95% accuracy
        if epoch_train_acc >= accuracy_threshold:
            print(
                f"Reached {accuracy_threshold*100:.1f}% accuracy at epoch {epoch+1}. Early stopping."
            )
            break

    return training_history


# Export model for deployment
def export_model(model, model_path="lightweight_tablebase_classifier"):
    """Export the model to lightweight formats for deployment"""
    # Save the PyTorch model
    print("Saving PyTorch model...")
    torch.save(model.state_dict(), f"{model_path}.pth")
    print(f"Saved PyTorch model to {model_path}.pth")

    # Convert to TorchScript for C++ deployment
    try:
        print("Converting to TorchScript...")
        model.eval()  # Set to evaluation mode
        example_input = torch.randn(1, 65)
        scripted_model = torch.jit.script(model)
        scripted_model.save(f"{model_path}.pt")
        print(f"Saved TorchScript model to {model_path}.pt")
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    # Try to export to ONNX if available
    try:
        print("Exporting to ONNX format...")
        model.eval()  # Ensure model is in eval mode
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
    except Exception as e:
        print(f"ONNX export failed: {e}")

    print("Model export complete!")


def main():
    # Hyperparameters
    input_size = 65  # 64 squares + 1 for side to move
    num_classes = 3  # Loss, Draw, Win
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 100  # Reduced epochs as the optimized model converges faster

    # Path to tablebase directory
    tablebase_dir = "/Users/arunim/Documents/github/tbcompress/Syzygy345_WDL"

    # Cache file to avoid reprocessing tablebase files
    cache_file = "/Users/arunim/Documents/github/tbcompress/tablebase_cache.pt"

    # Create dataset using all real tablebase data
    print("Creating dataset from all tablebases...")
    dataset = SyzygyTablebaseDataset(
        tablebase_dir=tablebase_dir,
        max_positions=50000,  # Increased to capture more positions
        cache_file=cache_file,
    )

    # Create data loader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the optimized model
    model = OptimizedTablebaseClassifier(input_size, num_classes)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train the model with optimized training function
    print("Starting training with optimized classifier...")
    start_time = time.time()

    history = train_optimized_classifier(
        model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(
        f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)!"
    )
    print(f"Final training accuracy: {max(history['train_acc']):.4f}")

    # Export model for deployment
    print("\nExporting model for deployment...")
    export_model(model)

    # Optional: Compare with legacy model
    print("\nFor comparison, here's the parameter count of the legacy model:")
    legacy_model = TablebaseClassifier(input_size, 256, num_classes)
    print(
        f"Legacy model parameters: {sum(p.numel() for p in legacy_model.parameters()):,}"
    )


if __name__ == "__main__":
    main()
