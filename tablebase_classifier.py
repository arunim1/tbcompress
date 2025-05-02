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


# Define a neural network for tablebase classification
class TablebaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TablebaseClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


# Function to convert a chess position to feature vector
def board_to_feature_vector(board):
    """Convert a chess.Board to a feature vector for the neural network."""
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

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = piece_values[piece.piece_type]
            if piece.color == chess.BLACK:
                value = -value
            feature_vector[square] = value

    # Extra features for side to move
    # If white to move, add a 1 at the end, else add a -1
    side_to_move = 1.0 if board.turn == chess.WHITE else -1.0

    # Combine features
    combined_features = np.append(feature_vector, side_to_move)
    return combined_features


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
            data = torch.load(cache_file)
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


# Training function
def train_tablebase_classifier(
    model, train_loader, criterion, optimizer, num_epochs=500
):
    # if cuda, use cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration")
    elif torch.backends.mps.is_available():
        # Check if MPS (Metal Performance Shaders) is available for Mac Silicon GPU acceleration
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for Mac Silicon GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, falling back to CPU")

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

    return training_history


def main():
    # Hyperparameters
    input_size = 65  # 64 squares + 1 for side to move
    hidden_size = 256
    num_classes = 3  # Loss, Draw, Win
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 500

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

    # Initialize the model
    model = TablebaseClassifier(input_size, hidden_size, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Starting training...")
    start_time = time.time()

    history = train_tablebase_classifier(
        model, train_loader, criterion, optimizer, num_epochs
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(
        f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)!"
    )
    print(f"Final training accuracy: {max(history['train_acc']):.4f}")


if __name__ == "__main__":
    main()
