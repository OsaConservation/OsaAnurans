"""Read explicit embedding rows without loading complete recordings into RAM."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from ..records import EmbeddingRow


class EmbeddingStore:
    """Read vectors from .npy files while reusing recently accessed matrices.

    Rules:
    - Memory-map files instead of copying whole matrices into RAM.
    - Retain at most max_open matrices, default eight.
    - Copy requested rows to float32 and validate them before returning.
    """

    def __init__(self, max_open: int = 8):
        """Set the number of recently used matrix mappings to retain."""
        self.max_open = max_open
        self._arrays: OrderedDict[Path, np.ndarray] = OrderedDict()

    def read(self, row: EmbeddingRow) -> np.ndarray:
        """Return one finite, nonempty vector from its file and row index.

        Unhappy paths:
        - Report missing/unreadable files with the affected window ID.
        - Reject invalid row indices, matrix shapes, or vector values.
        """
        path = row.path.resolve()
        array = self._arrays.pop(path, None)
        if array is None:
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"Cannot read embedding artifact for window {row.window_id}: {path} ({error})"
                ) from error
        # Moving this file to the end makes it the most recently used matrix.
        self._arrays[path] = array

        while len(self._arrays) > self.max_open:
            self._arrays.popitem(last=False)

        if array.ndim != 2 or not 0 <= row.row_index < len(array):
            raise ValueError(f"Invalid embedding location: {path}, row {row.row_index}")

        vector = np.array(array[row.row_index], dtype=np.float32, copy=True)
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError(f"Empty or nonfinite embedding for window {row.window_id}")
        return vector

    def read_batch(self, rows: list[EmbeddingRow]) -> np.ndarray:
        """Stack requested vectors in input order; reject an empty batch."""
        if not rows:
            raise ValueError("Cannot read an empty embedding batch")
        return np.stack([self.read(row) for row in rows])


class ProbeDataset(Dataset):
    """Expose window vectors and integer targets to PyTorch/bacpipe training.

    Rules:
    - rows[i] and targets[i] describe the same example.
    - Load each vector on demand through EmbeddingStore.
    """

    def __init__(self, rows: list[EmbeddingRow], targets: list[int]):
        """Require one integer class target for each indexed embedding row."""
        if len(rows) != len(targets):
            raise ValueError("Embedding rows and labels must have equal lengths")
        self.rows = rows
        self.targets = targets
        self.store = EmbeddingStore()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.store.read(self.rows[index]), self.targets[index]
