# /// script
# dependencies = [
#     "torch",
#     "numpy",
#     "python-chess",
#     "tqdm",
# ]
# ///

"""
Test script for the StreamingTablebaseDataset class
"""
import os
import sys
from tbcompress.streaming_dataset import StreamingTablebaseDataset
import torch
from torch.utils.data import DataLoader

def main():
    # Find Syzygy tablebase directory
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
    
    try:
        # Create a streaming dataset with a small cache for quick testing
        print(f"Creating streaming dataset for {os.path.basename(test_file)}...")
        dataset = StreamingTablebaseDataset(
            rtbw_file=test_file,
            tablebase_dir=syzygy_dir,
            cache_size=50,  # Small cache for testing
            max_positions=200  # Limit positions for quick testing
        )
        
        # Test retrieving positions from the dataset
        print("\nTesting position retrieval:")
        for i in range(10):
            features, label = dataset[i]
            print(f"Position {i}: shape={features.shape}, WDL class={label.item()}")
        
        # Test using the dataset with DataLoader
        print("\nTesting with DataLoader:")
        batch_size = 16
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,  # No need to shuffle since positions are generated on-the-fly
            num_workers=0  # Important: don't use multiple workers with streaming dataset
        )
        
        # Test running through a few batches
        print(f"Processing {batch_size} batches...")
        for i, (batch_features, batch_labels) in enumerate(dataloader):
            if i >= 3:  # Just test a few batches
                break
                
            print(f"Batch {i}: features={batch_features.shape}, labels={batch_labels.shape}")
            
            # Print distribution of WDL values in this batch
            values, counts = torch.unique(batch_labels, return_counts=True)
            distribution = {val.item(): count.item() for val, count in zip(values, counts)}
            print(f"  WDL distribution: {distribution}")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Error testing streaming dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
