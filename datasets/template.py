"""Starting point for a new environment's dataset.

Copy this file, implement the three methods, and point a config's
`dataset._target_` at the class. The slicer and the embedding cache work off
this interface alone.
"""

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import TensorDataset

from datasets.core import TrajectoryDataset
from datasets.types import Sample


class YourTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)

    def get_seq_length(self, idx: int) -> int:
        """Number of real (unpadded) timesteps in episode `idx`."""
        raise NotImplementedError

    def get_frames(self, idx: int, frames: Sequence[int]) -> Sample:
        """Return the requested timesteps of episode `idx`.

        Every tensor in the returned Sample has time on axis 0, in the order
        `frames` asks for. Do work proportional to len(frames), not to the
        episode: this is called once per training sample.

        `obs` is T V C H W, scaled to 0..1. `action` is T A. `goal` is
        whatever the policy conditions on, or a placeholder. `state` holds
        proprioception if the environment records it.
        """
        raise NotImplementedError
        # return {"obs": obs / 255.0, "action": actions, "goal": goal}

    def __getitem__(self, idx: int) -> Sample:
        return self.get_frames(idx, range(self.get_seq_length(idx)))
