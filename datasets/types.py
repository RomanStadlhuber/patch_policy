"""Shared shapes for the data the loaders pass around.

These are hints, not validators. They exist so a reader can tell what a
`get_frames` call returns without tracing tuple positions through four
dataset classes and the slicer.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Sequence

try:  # TypedDict moved into typing in 3.8, but keep the fallback explicit
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict

import torch


class Sample(TypedDict, total=False):
    """One slice of a trajectory, keyed by stream name.

    `obs` and `action` are always present. `goal` is present for the
    goal-conditioned environments and carries a placeholder elsewhere.
    `state` is proprioception, added by datasets that record it.
    """

    obs: torch.Tensor
    action: torch.Tensor
    goal: torch.Tensor
    state: torch.Tensor


#: streams that follow the observation window; everything else follows the
#: action window, which extends past it
OBS_WINDOW_STREAMS = ("obs", "goal", "state")


class StreamSpec(NamedTuple):
    """How one stream of an embedding cache is stored and addressed."""

    name: str
    role: str  # observation | label | goal | proprio
    dtype: str  # numpy dtype name, e.g. "float16"
    shape: Sequence[int]  # per element, without the leading axis
    storage: str  # disk | ram
    indexing: str  # frame | episode
    file: str


class StreamEntry(TypedDict):
    """One `streams` entry as it appears in manifest.json."""

    role: str
    file: str
    storage: str
    indexing: str
    dtype: str
    shape: List[int]


class EncoderRef(TypedDict):
    """Which encoder produced a cache, recorded for the reader, not for lookup."""

    hub_repo: str
    name: str
    feature_key: str


class Manifest(TypedDict):
    """Top level of manifest.json."""

    version: int
    encoder: EncoderRef
    dataset: str
    num_episodes: int
    num_frames: int
    streams: Dict[str, StreamEntry]
