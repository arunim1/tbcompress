import numpy as np
import torch
from torch.utils.data import IterableDataset
import tb_stream  # the wheel built by maturin


class SyzygyIterable(IterableDataset):
    def __init__(self, tb_path: str, seed: int = 42):
        super().__init__()
        self._it = tb_stream.make_tb_stream(tb_path, seed)

    def __iter__(self):
        for planes, wdl in self._it:
            # np.uint8 → float32 torch tensor in [0,1]
            yield (
                torch.from_numpy(np.asarray(planes, dtype=np.float32)),
                torch.tensor(wdl, dtype=torch.float32),
            )
