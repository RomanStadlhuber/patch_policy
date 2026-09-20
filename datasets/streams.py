"""Per-stream storage for a precomputed embedding cache.

A cache directory holds one file per stream plus a manifest describing them:

    manifest.json   stream table (name, role, dtype, shape, storage, indexing)
    index.pt        offsets and seq_lengths, shared by every frame-indexed stream
    obs.dat         big, read by byte offset
    action.pt       small, loaded whole

Streams share one row numbering, so `offsets[e] + f` addresses episode `e`'s
frame `f` in every frame-indexed stream. Adding proprioception adds a file and
a manifest entry; nothing about the existing files changes.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from datasets.types import EncoderRef, Manifest, StreamEntry, StreamSpec

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.pt"


def _spec(name: str, entry: StreamEntry) -> StreamSpec:
    return StreamSpec(
        name=name,
        role=entry["role"],
        dtype=entry["dtype"],
        shape=tuple(entry["shape"]),
        storage=entry["storage"],
        indexing=entry["indexing"],
        file=entry["file"],
    )


class StreamStore:
    """Reads the streams of one cache directory."""

    def __init__(self, root: "os.PathLike") -> None:
        self.root = Path(root)
        with open(self.root / MANIFEST_NAME) as fh:
            self.manifest: Manifest = json.load(fh)
        if self.manifest["version"] != MANIFEST_VERSION:
            raise ValueError(
                f"{self.root/MANIFEST_NAME}: version {self.manifest['version']}, "
                f"expected {MANIFEST_VERSION}"
            )
        self.specs: Dict[str, StreamSpec] = {
            name: _spec(name, entry)
            for name, entry in self.manifest["streams"].items()
        }

        index = torch.load(self.root / INDEX_NAME, weights_only=False)
        self.offsets: List[int] = index["offsets"]
        self.seq_lengths: List[int] = index["seq_lengths"]

        # small streams are resident; big ones are read per sample
        self._resident: Dict[str, torch.Tensor] = {
            name: torch.load(self.root / spec.file, weights_only=False)
            for name, spec in self.specs.items()
            if spec.storage == "ram"
        }
        self._strides: Dict[str, int] = {
            name: int(np.prod(spec.shape)) * np.dtype(spec.dtype).itemsize
            for name, spec in self.specs.items()
            if spec.storage == "disk"
        }
        # descriptors are per process: dataloader workers are forked
        self._fds: Dict[str, int] = {}
        self._fd_pid: Optional[int] = None

    @staticmethod
    def exists(root: "os.PathLike") -> bool:
        root = Path(root)
        return (root / MANIFEST_NAME).is_file() and (root / INDEX_NAME).is_file()

    @property
    def num_episodes(self) -> int:
        return len(self.seq_lengths)

    def seq_length(self, episode: int) -> int:
        return self.seq_lengths[episode]

    def _descriptor(self, name: str) -> int:
        pid = os.getpid()
        if self._fd_pid != pid:
            self._fds = {}
            self._fd_pid = pid
        if name not in self._fds:
            path = self.root / self.specs[name].file
            self._fds[name] = os.open(str(path), os.O_RDONLY)
        return self._fds[name]

    def _read_disk(self, name: str, rows: np.ndarray) -> torch.Tensor:
        spec = self.specs[name]
        dtype = np.dtype(spec.dtype)
        stride = self._strides[name]
        fd = self._descriptor(name)

        chunks: List[np.ndarray] = []
        start = 0
        while start < len(rows):
            # consecutive rows are contiguous on disk, so one read serves them
            end = start + 1
            while end < len(rows) and rows[end] == rows[end - 1] + 1:
                end += 1
            raw = os.pread(fd, (end - start) * stride, int(rows[start]) * stride)
            chunks.append(np.frombuffer(raw, dtype=dtype))
            start = end
        flat = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        # np.frombuffer is read-only; torch needs a buffer it can own
        return torch.from_numpy(flat.reshape(len(rows), *spec.shape).copy())

    def read(self, name: str, episode: int, frames: np.ndarray) -> torch.Tensor:
        """Read `frames` of `episode` from one stream, however it is stored."""
        spec = self.specs[name]
        if spec.indexing == "episode":
            # one row per episode, repeated to match the requested frames
            row = np.full(len(frames), episode, dtype=np.int64)
        else:
            row = frames + self.offsets[episode]
        if spec.storage == "ram":
            return self._resident[name][torch.as_tensor(row)]
        return self._read_disk(name, row)

    def all_rows(self, name: str) -> torch.Tensor:
        """Every row of a resident stream, unpadded episodes concatenated."""
        spec = self.specs[name]
        if spec.storage != "ram":
            raise ValueError(f"stream {name!r} is on disk; read it per sample")
        return self._resident[name]


class StreamWriter:
    """Builds a cache directory one stream at a time."""

    def __init__(
        self,
        root: "os.PathLike",
        encoder: EncoderRef,
        dataset_name: str,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self.dataset_name = dataset_name
        self.entries: Dict[str, StreamEntry] = {}
        self._handles: Dict[str, Any] = {}

    def disk_stream(
        self, name: str, role: str, dtype: str, indexing: str = "frame"
    ) -> None:
        """Open a stream written by appending arrays in row order."""
        file = f"{name}.dat"
        self._handles[name] = open(self.root / file, "wb")
        self.entries[name] = {
            "role": role,
            "file": file,
            "storage": "disk",
            "indexing": indexing,
            "dtype": dtype,
            "shape": [],  # filled by the first append
        }

    def append(self, name: str, rows: np.ndarray) -> None:
        entry = self.entries[name]
        if not entry["shape"]:
            entry["shape"] = list(rows.shape[1:])
        self._handles[name].write(rows.astype(entry["dtype"]).tobytes())

    def ram_stream(
        self,
        name: str,
        role: str,
        rows: torch.Tensor,
        indexing: str = "frame",
    ) -> None:
        """Store a stream small enough to load whole."""
        file = f"{name}.pt"
        torch.save(rows, self.root / file)
        self.entries[name] = {
            "role": role,
            "file": file,
            "storage": "ram",
            "indexing": indexing,
            "dtype": str(rows.dtype).replace("torch.", ""),
            "shape": list(rows.shape[1:]),
        }

    def finalize(self, seq_lengths: Sequence[int], offsets: Sequence[int]) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles = {}
        torch.save(
            {"seq_lengths": list(seq_lengths), "offsets": list(offsets)},
            self.root / INDEX_NAME,
        )
        manifest: Manifest = {
            "version": MANIFEST_VERSION,
            "encoder": self.encoder,
            "dataset": self.dataset_name,
            "num_episodes": len(seq_lengths),
            "num_frames": int(sum(seq_lengths)),
            "streams": self.entries,
        }
        with open(self.root / MANIFEST_NAME, "w") as fh:
            json.dump(manifest, fh, indent=2)


def migrate_legacy_cache(
    root: "os.PathLike",
    encoder: EncoderRef,
    dataset_name: str,
    obs_dtype: str = "float16",
) -> None:
    """Give a pre-manifest cache a manifest, without rewriting obs.dat.

    The first caches stored obs.dat plus a meta.pt holding offsets, lengths and
    a per-episode list of the remaining tensors. The frame layout is already
    what the manifest describes, so only the sidecars are rebuilt.
    """
    root = Path(root)
    meta = torch.load(root / "meta.pt", weights_only=False)
    seq_lengths: List[int] = meta["seq_lengths"]
    offsets: List[int] = meta["offsets"]
    rest = meta["rest"]

    writer = StreamWriter(root, encoder, dataset_name)
    # obs.dat is already in place; describe it rather than copy it
    writer.entries["obs"] = {
        "role": "observation",
        "file": "obs.dat",
        "storage": "disk",
        "indexing": "frame",
        "dtype": obs_dtype,
        "shape": list(meta["frame_shape"]),
    }
    # positional order was (action, goal), per "assuming goal comes last"
    names = ("action", "goal")
    roles = {"action": "label", "goal": "goal"}
    per_episode: Dict[str, List[torch.Tensor]] = {}
    for episode in rest:
        items = episode.items() if isinstance(episode, dict) else zip(names, episode)
        for name, value in items:
            per_episode.setdefault(name, []).append(value)
    for name, values in per_episode.items():
        writer.ram_stream(name, roles.get(name, "label"), torch.cat(values, dim=0))
    writer.finalize(seq_lengths, offsets)
