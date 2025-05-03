# /// script
# dependencies = [
#     "torch",
#     "numpy",
#     "python-chess",
#     "tqdm",
# ]
# ///

"""
Test script for the StreamingTablebaseDataset
"""
import os
import sys
from tbcompress.data import StreamingTablebaseDataset


def main():
    # Use Syzygy345_WDL as both the source for rtbw files and as the tablebase directory
    rtbw_dir = os.path.join(os.path.dirname(__file__), "Syzygy345_WDL")
    tablebase_dir = rtbw_dir  # Use the same directory for tablebase files

    # Find simple .rtbw files with few pieces (like KvK, KQvK, etc.)
    print(f"Looking for .rtbw files in {rtbw_dir}...")
    simple_pieces = ["KvK", "KQvK", "KRvK", "KBvK", "KNvK", "KPvK"]
    rtbw_files = []
    
    try:
        all_files = os.listdir(rtbw_dir)
        rtbw_files = [f for f in all_files if f.endswith(".rtbw")]
        
        # Try to find a simple file first
        for simple in simple_pieces:
            for f in rtbw_files:
                if simple in f:
                    rtbw_files = [f]  # Use this file
                    print(f"Found simple tablebase file: {f}")
                    break
            if rtbw_files and len(rtbw_files) == 1:
                break
    except FileNotFoundError:
        print(f"Error: Directory {rtbw_dir} not found")
        sys.exit(1)
        
    if not rtbw_files:
        print(f"Error: No .rtbw files found in {rtbw_dir}")
        sys.exit(1)

    rtbw_file = os.path.join(rtbw_dir, rtbw_files[0])
    print(f"Using sample file: {rtbw_file}")

    # Create the streaming dataset with a small cache
    try:
        dataset = StreamingTablebaseDataset(
            rtbw_file=rtbw_file,
            tablebase_dir=tablebase_dir,
            cache_size=1000,  # Small cache for quick testing
            max_positions=10000,  # Limit for testing
        )

        print(f"Successfully created StreamingTablebaseDataset")
        print(f"Estimated dataset size: {len(dataset)} positions")

        # Test retrieving some data
        print(f"Testing data retrieval:")
        for i in range(5):
            position, wdl = dataset[i]
            print(f"  Position {i}: shape={position.shape}, WDL={wdl.item()}")

        print("All tests passed!")
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
