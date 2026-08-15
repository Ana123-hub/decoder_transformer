import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MemmapDataset(Dataset):
    """
    Zero-RAM-overhead PyTorch Dataset streaming token blocks directly 
    from binary files using numpy.memmap.
    """
    def __init__(self, bin_path: str, block_size: int):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary file not found at {bin_path}")
        
        self.block_size = block_size
        # Stream uint16 tokens directly from disk
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        # Total available starting positions
        self.num_samples = len(self.data) - block_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # Extract sequence of length block_size + 1
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        
        # x is sequence 0..N-1, y is sequence shifted by 1 position: 1..N
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def get_dataloader(bin_path: str, block_size: int, batch_size: int, shuffle: bool = True):
    """Utility helper function to return DataLoader for training/validation."""
    dataset = MemmapDataset(bin_path, block_size)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0,      # memmap is fast enough in-process without multiprocessing overhead
        pin_memory=True
    )