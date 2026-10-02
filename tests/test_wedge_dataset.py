"""Wedge dataset and chunked-action slicing on a tiny synthetic export."""

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from datasets.core import (
    FileEmbeddingDataset,
    TrajectoryEmbeddingDataset,
    TrajectorySlicerDataset,
)
from datasets.wedge import WedgeDataset

H, A, SIZE = 16, 10, 8
SEQ_LENGTHS = [5, 3, 7]
WINDOW, ACTION_WINDOW = 2, 15


def action_row(episode: int, t: int) -> torch.Tensor:
    """A (H, A) row that encodes its episode and frame, so a test can tell rows apart."""
    base = 1000.0 * (episode + 1) + t
    return base + torch.arange(H * A, dtype=torch.float32).reshape(H, A) * 1e-3


def write_export(root: Path, seq_lengths: List[int] = SEQ_LENGTHS) -> Path:
    (root / "obses").mkdir(parents=True)
    t_max = max(seq_lengths)
    actions = torch.zeros(len(seq_lengths), t_max, H, A)
    for e, T in enumerate(seq_lengths):
        frames = np.zeros((T, SIZE, SIZE, 3), dtype=np.uint8)
        frames[:] = np.arange(T, dtype=np.uint8)[:, None, None, None] * 10 + e
        torch.save(frames, root / "obses" / f"episode_{e:05d}.pth")
        for t in range(T):
            actions[e, t] = action_row(e, t)
    torch.save(actions, root / "actions.pth")
    with open(root / "seq_lengths.pkl", "wb") as f:
        pickle.dump(seq_lengths, f)
    (root / "export.yaml").write_text("format_version: 3\n")
    return root


class MeanEncoder(torch.nn.Module):
    """(T, V, C, H, W) -> (T, V, P=1, E=3): the per-channel mean, deterministic."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-1, -2)).unsqueeze(-2)


@pytest.fixture
def export(tmp_path: Path) -> Path:
    return write_export(tmp_path / "wedge_export")


@pytest.mark.parametrize("prefetch", [True, False])
def test_dataset_shapes(export: Path, prefetch: bool) -> None:
    ds = WedgeDataset(export, prefetch=prefetch)
    assert len(ds) == 3
    sample = ds.get_frames(2, [1, 2], stream_frames={"action": [4]})
    assert sample["obs"].shape == (2, 1, 3, SIZE, SIZE)
    assert sample["obs"].dtype == torch.float32
    assert 0.0 <= sample["obs"].min() and sample["obs"].max() <= 1.0
    assert sample["action"].shape == (1, H, A)
    assert torch.equal(sample["action"][0], action_row(2, 4))
    assert sample["goal"].shape == (2, 1, 1, 1)
    assert ds.get_all_actions().shape == (sum(SEQ_LENGTHS), H, A)


def expected_slices(seq_lengths: List[int]) -> List[tuple]:
    out = []
    for e, T in enumerate(seq_lengths):
        out += [(e, 0, end + 1) for end in range(WINDOW - 1)]
        out += [(e, s, s + WINDOW) for s in range(T - WINDOW + 1)]
    return out


def check_slicer(data) -> None:
    slicer = TrajectorySlicerDataset(
        data, window=WINDOW, action_window=ACTION_WINDOW, chunked_actions=True
    )
    slices = expected_slices(SEQ_LENGTHS)
    assert slicer.slices == slices
    for i, (e, start, end) in enumerate(slices):
        sample = slicer[i]
        assert sample["action"].shape == (H, A), (i, sample["action"].shape)
        # the chunk is the row of the window's last frame
        assert torch.allclose(sample["action"].float(), action_row(e, end - 1)), (e, start, end)
        assert sample["obs"].shape[0] == WINDOW
        assert sample["goal"].shape[0] == WINDOW


def test_slicer_raw(export: Path) -> None:
    check_slicer(WedgeDataset(export, prefetch=False))


def test_slicer_short_first_window_repeats_obs(export: Path) -> None:
    slicer = TrajectorySlicerDataset(
        WedgeDataset(export), window=WINDOW, action_window=ACTION_WINDOW, chunked_actions=True
    )
    first = slicer[0]  # (0, 0, 1): frame 0 only
    assert torch.equal(first["obs"][0], first["obs"][1])


def test_slicer_ram_embedding(export: Path) -> None:
    emb = TrajectoryEmbeddingDataset(MeanEncoder(), WedgeDataset(export), device="cpu")
    check_slicer(emb)
    assert emb.get_all_actions().shape == (sum(SEQ_LENGTHS), H, A)


def test_slicer_file_embedding(export: Path, tmp_path: Path) -> None:
    raw = WedgeDataset(export)
    for build in (True, False):  # write the cache, then reuse it
        emb = FileEmbeddingDataset(
            MeanEncoder(), raw, tmp_path / "cache", "key", dtype=np.float32
        )
        assert emb.store.specs["action"].shape == (H, A)
        assert torch.equal(emb.get_all_actions(), raw.get_all_actions())
        check_slicer(emb)
        # same features as the RAM path
        ram = TrajectoryEmbeddingDataset(MeanEncoder(), raw, device="cpu")
        for e, T in enumerate(SEQ_LENGTHS):
            got = emb.get_frames(e, range(T))
            assert torch.allclose(got["obs"], ram.data[e]["obs"])
            assert torch.equal(got["action"], ram.data[e]["action"])


def test_unchunked_slicer_unchanged(export: Path) -> None:
    """Without the flag the slicer still slices a window of rows (Cube behavior)."""
    slicer = TrajectorySlicerDataset(
        WedgeDataset(export),
        window=WINDOW,
        action_window=3,
        vqbet_get_future_action_chunk=False,
    )
    assert slicer[0]["action"].shape == (WINDOW + 3 - 1, H, A)


def test_normalizer_per_position(export: Path) -> None:
    """A last_n_dims=2 normalizer maps each (position, dim) range to +/-0.5 and inverts."""
    from utils.normalizer import LinearNormalizer

    actions = WedgeDataset(export).get_all_actions()
    norm = LinearNormalizer()
    norm.fit(actions, last_n_dims=2, output_min=-0.5, output_max=0.5)
    assert norm.params_dict["_default"]["scale"].shape == (H * A,)
    batch = actions[:4]
    n = norm.normalize(batch)
    assert n.shape == batch.shape
    full = norm.normalize(actions)
    assert torch.allclose(full.amin(0), torch.full((H, A), -0.5), atol=1e-4)
    assert torch.allclose(full.amax(0), torch.full((H, A), 0.5), atol=1e-4)
    assert torch.allclose(norm.unnormalize(n), batch, atol=1e-2)
