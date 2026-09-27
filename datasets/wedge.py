"""Wedge grasp demos exported by supermanipulation's policy exporter.

Layout (the Cube layout with chunked actions):

    obses/episode_{i:05d}.pth   uint8 numpy (T, 224, 224, 3), one view
    actions.pth                 float32 torch (N, T_max, H, A), zero-padded past each T
    seq_lengths.pkl             list of per-episode T
    export.yaml                 source run, git commit, format version, dims

Row `t` of an episode's actions is the whole chunk for the window ending at frame `t`
(H = 16 heel-frame deltas from the heel pose at `t`, see `supermanipulation.wedge.actions`),
so every stream stays per frame and `TrajectorySlicerDataset(chunked_actions=True)` picks
one row instead of slicing a window of actions.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Sequence

import einops
import numpy as np
import torch

from datasets.core import TrajectoryDataset, frames_for
from datasets.types import Sample, StreamFrames


class WedgeDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: str,
        subset_fraction: Optional[float] = None,
        prefetch: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Load actions and lengths; frames are loaded here only when `prefetch` is set.

        Args:
            subset_fraction: Keep only the first fraction of the episodes.
            prefetch: Hold every episode's raw frames in RAM instead of reading them per sample.
        """
        self.data_directory = Path(data_directory)
        self.actions: torch.Tensor = torch.load(
            self.data_directory / "actions.pth", weights_only=False
        )
        with open(self.data_directory / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths: List[int] = [int(t) for t in pickle.load(f)]
        if self.actions.ndim != 4:
            raise ValueError(
                f"{self.data_directory}/actions.pth: expected (N, T_max, H, A), "
                f"got {tuple(self.actions.shape)}"
            )
        if len(self.seq_lengths) != self.actions.shape[0]:
            raise ValueError(
                f"{self.data_directory}: {len(self.seq_lengths)} seq_lengths, "
                f"{self.actions.shape[0]} action episodes"
            )

        self.subset_fraction = subset_fraction
        if self.subset_fraction:
            assert 0 < self.subset_fraction <= 1
            n = int(len(self.seq_lengths) * self.subset_fraction)
        else:
            n = len(self.seq_lengths)
        self.actions = self.actions[:n].float()
        self.seq_lengths = self.seq_lengths[:n]
        for i, T in enumerate(self.seq_lengths):
            self.actions[i, T:] = 0  # redo zero padding

        self.prefetch = prefetch
        self.obses: Optional[List[np.ndarray]] = None
        if self.prefetch:
            self.obses = [self._load_obs(i) for i in range(n)]

    def _load_obs(self, idx: int) -> np.ndarray:
        obs = torch.load(
            str(self.data_directory / "obses" / f"episode_{idx:05d}.pth"),
            weights_only=False,
        )
        return obs.numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)

    def get_seq_length(self, idx: int) -> int:
        return self.seq_lengths[idx]

    def get_all_actions(self) -> torch.Tensor:
        """Every episode's chunk rows, unpadded: `(sum T, H, A)`."""
        return torch.cat(
            [self.actions[i, :T] for i, T in enumerate(self.seq_lengths)], dim=0
        )

    def get_frames(
        self,
        idx: int,
        frames: Sequence[int],
        stream_frames: Optional[StreamFrames] = None,
    ) -> Sample:
        frames = list(frames)
        episode = self.obses[idx] if self.obses is not None else self._load_obs(idx)
        obs = einops.rearrange(episode[frames], "T H W C -> T 1 C H W")  # 1 view
        obs = torch.from_numpy(obs.astype(np.float32) / 255.0)
        act = self.actions[idx, list(frames_for("action", frames, stream_frames))]  # k H A
        dummy_goal = torch.ones([obs.shape[0], 1, 1, 1])  # T V P E, as Cube
        return {"obs": obs, "action": act, "goal": dummy_goal}

    def __getitem__(self, idx: int) -> Sample:
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self) -> int:
        return len(self.seq_lengths)
