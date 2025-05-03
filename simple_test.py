# /// script
# dependencies = [
#     "torch",
#     "numpy",
#     "python-chess",
#     "tqdm",
# ]
# ///

"""
Simple test script for the TablebaseDataset to be used with Syzygy files
"""
import os
import sys
import torch
import chess
import chess.syzygy
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Define feature vector conversion function
def board_to_feature_vector(board):
    """Convert a chess.Board to a feature vector for the neural network"""
    feature_vector = np.zeros(64, dtype=np.float32)
    
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
        chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
    }
    
    # Populate non-empty squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = piece_values[piece.piece_type]
            if piece.color == chess.BLACK:
                value = -value
            feature_vector[square] = value
            
    # Add side to move
    side_to_move = 1.0 if board.turn == chess.WHITE else -1.0
    combined_features = np.append(feature_vector, side_to_move)
    return combined_features

# Simple dataset for testing
class SimpleTablebaseDataset(Dataset):
    """Basic dataset for testing with Syzygy tablebases"""
    
    def __init__(self, rtbw_file, tablebase_dir, max_positions=1000):
        """
        Create a simple dataset from a specific .rtbw tablebase file
        """
        self.rtbw_file = rtbw_file
        self.tablebase_dir = tablebase_dir
        self.max_positions = max_positions
        
        # Initialize tablebase
        self.tablebase = chess.syzygy.open_tablebase(tablebase_dir)
        
        # Extract material configuration from filename
        self.material = os.path.basename(rtbw_file).split(".")[0]
        print(f"Material configuration: {self.material}")
        
        # Generate a small test dataset
        self.features = []
        self.labels = []
        self._generate_test_positions()
    
    def _generate_test_positions(self):
        """Generate a small number of test positions"""
        print(f"Generating test positions for {self.material}...")
        
        # Simple KvK or similar with only kings
        if self.material in ["KvK", "KQvK", "KRvK", "KBvK", "KNvK", "KPvK"]:
            positions_added = 0
            
            # Try king placements systematically
            for wk_square in range(0, 64, 4):  # Sample some white king positions
                for bk_square in range(0, 64, 4):  # Sample some black king positions
                    if chess.square_distance(wk_square, bk_square) <= 1:
                        continue  # Kings too close
                    
                    # Create a basic board with two kings
                    board = chess.Board(fen=None)
                    board.clear_board()
                    board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                    board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                    
                    # If there's an additional piece in the material name
                    if "Q" in self.material:
                        # Try a few queen placements
                        for q_square in [8, 36, 55]:  # Arbitrary placements
                            if q_square == wk_square or q_square == bk_square:
                                continue
                            
                            test_board = board.copy()
                            test_board.set_piece_at(q_square, chess.Piece(chess.QUEEN, chess.WHITE))
                            
                            # Try white to move
                            test_board.turn = chess.WHITE
                            if self._try_add_position(test_board):
                                positions_added += 1
                            
                            # Try black to move
                            test_board.turn = chess.BLACK
                            if self._try_add_position(test_board):
                                positions_added += 1
                    else:
                        # Just kings
                        # Try white to move
                        board.turn = chess.WHITE
                        if self._try_add_position(board):
                            positions_added += 1
                        
                        # Try black to move
                        board.turn = chess.BLACK
                        if self._try_add_position(board):
                            positions_added += 1
                    
                    if positions_added >= self.max_positions:
                        break
                
                if positions_added >= self.max_positions:
                    break
            
            print(f"Generated {positions_added} test positions")
    
    def _try_add_position(self, board):
        """Try to add a position to the dataset, checking if it's valid"""
        if not board.is_valid() or board.is_check():
            return False
            
        try:
            # Get WDL value from tablebase
            wdl = self.tablebase.probe_wdl(board)
            
            # Convert WDL to target label (0=loss, 1=draw, 2=win)
            # If side to move is black, negate the WDL
            if not board.turn:
                wdl = -wdl
                
            # Map from [-2, 0, 2] to [0, 1, 2] (loss, draw, win)
            label = (wdl + 2) // 2
            
            # Add to dataset
            features = board_to_feature_vector(board)
            self.features.append(features)
            self.labels.append(label)
            return True
            
        except (ValueError, chess.syzygy.MissingTableError) as e:
            # Position not found in tablebase
            return False
    
    def __len__(self):
        """Return the number of positions in the dataset"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """Get a position by index"""
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )

def main():
    # Find Syzygy tablebase directory containing .rtbw files
    syzygy_dir = os.path.join(os.path.dirname(__file__), "Syzygy345_WDL")
    if not os.path.exists(syzygy_dir):
        print(f"Error: Syzygy directory {syzygy_dir} not found")
        sys.exit(1)
    
    # Find a simple tablebase file to test with
    simple_materials = ["KQvK.rtbw", "KRvK.rtbw", "KBvK.rtbw", "KNvK.rtbw", "KPvK.rtbw"]
    test_file = None
    
    for material in simple_materials:
        file_path = os.path.join(syzygy_dir, material)
        if os.path.exists(file_path):
            test_file = file_path
            print(f"Found test file: {test_file}")
            break
    
    if test_file is None:
        print(f"Error: No test files found in {syzygy_dir}")
        sys.exit(1)
    
    # Create and test the dataset
    max_test_positions = 100  # Small number for quick testing
    try:
        dataset = SimpleTablebaseDataset(
            rtbw_file=test_file,
            tablebase_dir=syzygy_dir,
            max_positions=max_test_positions
        )
        
        if len(dataset) == 0:
            print("Warning: Dataset is empty. Please check the tablebase file.")
            sys.exit(1)
        
        print(f"Dataset created successfully with {len(dataset)} positions")
        
        # Test accessing a few positions
        print("\nTesting data access:")
        for i in range(min(5, len(dataset))):
            features, label = dataset[i]
            print(f"Position {i}: shape={features.shape}, WDL class={label.item()}")
        
        # Test loading with DataLoader
        print("\nTesting DataLoader:")
        batch_size = 8
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for i, (batch_features, batch_labels) in enumerate(loader):
            if i >= 2:  # Just test first two batches
                break
                
            print(f"Batch {i}: features shape={batch_features.shape}, labels shape={batch_labels.shape}")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
