import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import tb_stream


class SyzygyIterable(IterableDataset):
    def __init__(self, tb_path: str, seed: int = 42, batch_size: int = 4096):
        super().__init__()
        self.tb_path = tb_path
        self.base_seed = seed
        self.batch_size = batch_size

    def __iter__(self):
        worker = get_worker_info()
        seed = self.base_seed + (worker.id if worker else 0)
        it = tb_stream.make_tb_stream(self.tb_path, seed, self.batch_size)
        for planes, labels in it:
            x = torch.from_numpy(np.asarray(planes, dtype=np.float32))
            y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
            yield x, y
