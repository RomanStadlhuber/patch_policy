import abc
import hashlib
import json
import os
import utils
import torch
import numpy as np
from pathlib import Path
from torch import default_generator, randperm
from torch.utils.data import Dataset, Subset
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from datasets.streams import StreamStore, StreamWriter, migrate_legacy_cache
from datasets.types import EncoderRef, Sample, StreamFrames


# Taken from python 3.5 docs
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns a Sample: {"obs": ..., "action": ..., ...}
        every tensor has time on axis 0, with the same length T
    """

    @abc.abstractmethod
    def get_seq_length(self, idx: int) -> int:
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_frames(
        self,
        idx: int,
        frames: Sequence[int],
        stream_frames: Optional[StreamFrames] = None,
    ) -> Sample:
        """
        Returns the frames from the idx-th trajectory at the specified frames.
        Used to speed up slicing.

        Every stream reads `frames`, except the streams named in
        `stream_frames`, which read their own list instead (see frames_for).
        """
        raise NotImplementedError


def frames_for(
    name: str, frames: Sequence[int], stream_frames: Optional[StreamFrames]
) -> Sequence[int]:
    """The frames stream `name` reads: its own list if it has one, else `frames`."""
    if stream_frames is None:
        return frames
    return stream_frames.get(name, frames)


class TrajectorySubset(TrajectoryDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx: int) -> int:
        return self.dataset.get_seq_length(self.indices[idx])

    def get_all_actions(self) -> torch.Tensor:
        return self.dataset.get_all_actions()

    def get_frames(
        self,
        idx: int,
        frames: Sequence[int],
        stream_frames: Optional[StreamFrames] = None,
    ) -> Sample:
        return self.dataset.get_frames(self.indices[idx], frames, stream_frames)


class TrajectorySlicerDataset(Dataset):
    """
    Slice a trajectory dataset into (overlapping) windows of `window` observations
    paired with an action chunk.

    dataset: a trajectory dataset that satisfies:
        dataset.get_seq_length(i) returns the length of sequence i
        dataset.get_frames(i, frames, stream_frames) -> Sample, each stream
            sliced to frames, or to its own list in stream_frames
        observations: Tensor[T, ...]
        actions: Tensor[T, ...]
    window: int
        number of observation timesteps in each slice
    action_window: int
        number of action timesteps to predict
    vqbet_get_future_action_chunk: bool = True
        if True, return only the action chunk following the observation window;
        otherwise return actions spanning the whole window plus the chunk
    transform: function (values) -> values
    pad_seq_length: bool = True
        pad actions at the end to ensure a fixed length instead of dropping short slices

    Goal conditioning is not handled here: any goal tensor is carried through as part
    of `*others` by the underlying dataset. The remaining arguments
    (future_conditional, min_future_sep, future_seq_len, only_sample_tail,
    use_libero_goal) are recorded on the instance but do not affect slicing;
    future_conditional only changes the value reported by get_seq_length.
    """

    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        action_window: int,
        vqbet_get_future_action_chunk: bool = True,
        future_conditional: bool = False,
        min_future_sep: int = 0,
        future_seq_len: Optional[int] = None,
        only_sample_tail: bool = False,
        transform: Optional[Callable] = None,
        use_libero_goal: bool = False,
        pad_seq_length: bool = True,  # pad actions at end to ensure fixed length
    ):
        if future_conditional:
            assert future_seq_len is not None, "must specify a future_seq_len"
        self.dataset = dataset
        self.window = window
        self.action_window = action_window
        self.vqbet_get_future_action_chunk = vqbet_get_future_action_chunk
        self.future_conditional = future_conditional
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.transform = transform
        self.slices = []
        self.use_libero_goal = use_libero_goal
        self.pad_seq_length = pad_seq_length

        min_seq_length = np.inf
        min_window_required = window + action_window - 1
        for i in range(len(self.dataset)):  # type: ignore
            T = self.dataset.get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)

            self.slices += [
                (i, 0, end + 1) for end in range(window - 1)
            ]  # slice indices follow convention [start, end)

            if self.pad_seq_length:
                if T - self.window >= 0:
                    self.slices += [
                        (i, start, start + self.window)
                        for start in range(T - self.window + 1)
                    ]
            else:
                if T - min_window_required < 0:
                    print(
                        f"Ignored short sequence #{i}: len={T}, window={min_window_required}"
                    )
                else:
                    self.slices += [
                        (i, start, start + self.window)
                        for start in range(T - min_window_required + 1)
                    ]

        if (not self.pad_seq_length) and (min_seq_length < min_window_required):
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        if self.future_conditional:
            return self.future_seq_len + self.window
        else:
            return self.window

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx: int) -> Sample:
        idx_episode, idx_start, idx_end = self.slices[idx]
        # fetch only the frames this slice needs instead of the whole episode:
        # dataset[idx_episode] reprocesses every frame of the episode, while
        # get_frames does work proportional to what's requested
        T = self.dataset.get_seq_length(idx_episode)
        # every stream but "action" reads the observation window, which ends at
        # `idx_end`; "action" also spans the chunk after it (-1 due to overlap
        # for 1 step between obs and act). Both start at `idx_start`, so frame k
        # is the same timestep in every stream.
        action_end = min(idx_end - 1 + self.action_window, T)
        sample = self.dataset.get_frames(
            idx_episode,
            range(idx_start, idx_end),
            stream_frames={"action": range(idx_start, action_end)},
        )
        act = sample["action"]

        values: Sample = {}
        # the first slices of an episode are shorter than the window: repeat
        # their first frame to fill it
        if idx_end - idx_start < self.window:
            act = utils.inference.repeat_start_to_length(
                act, self.window + self.action_window - 1, dim=0
            )
            for name, value in sample.items():
                if name == "action":
                    continue
                values[name] = utils.inference.repeat_start_to_length(
                    value, self.window, dim=0
                )
        else:
            for name, value in sample.items():
                if name == "action":
                    continue
                values[name] = value

        if self.vqbet_get_future_action_chunk:
            expected_len = self.action_window
            act = act[self.window - 1 :]
        else:
            expected_len = self.window + self.action_window - 1

        if act.shape[0] < expected_len:
            if self.pad_seq_length:
                act = utils.inference.repeat_end_to_length(act, expected_len, dim=0)
            else:
                raise ValueError(
                    f"Action chunk too short: {act.shape[0]} < {expected_len}, "
                    f"but pad_seq_length is False"
                )
        values["action"] = act

        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        return values


class TrajectoryEmbeddingDataset(TrajectoryDataset):
    def __init__(
        self,
        model,
        dataset: TrajectoryDataset,
        device="cpu",
        embed_goal=False,
        dtype=None,
    ):
        self.data = utils.inference.embed_trajectory_dataset(
            model,
            dataset,
            obs_only=False,
            device=device,
            embed_goal=embed_goal,
            dtype=dtype,
        )
        assert len(self.data) == len(dataset)
        # one Sample per episode, kept unpadded: padding to the longest episode
        # doubles the RAM and every read takes a whole episode anyway
        self.seq_lengths = [len(x["obs"]) for x in self.data]

    def get_seq_length(self, idx: int) -> int:
        return self.seq_lengths[idx]

    def get_all_actions(self) -> torch.Tensor:
        return torch.cat([episode["action"] for episode in self.data], dim=0)

    def get_frames(
        self,
        idx: int,
        frames: Sequence[int],
        stream_frames: Optional[StreamFrames] = None,
    ) -> Sample:
        return {
            name: value[frames_for(name, frames, stream_frames)]
            for name, value in self.data[idx].items()
        }

    def __getitem__(self, idx: int) -> Sample:
        # the stored tensors, not get_frames(range(...)): a range index copies the episode
        return dict(self.data[idx])

    def __len__(self) -> int:
        return len(self.seq_lengths)


class FileEmbeddingDataset(TrajectoryDataset):
    """Precomputed embeddings kept in a cache directory instead of in RAM.

    TrajectoryEmbeddingDataset holds every episode's features resident, which
    is ~30 GiB for Cube at fp32 and does not fit a 31 GB host. This writes each
    stream to its own file once and reads back only the rows a sample needs, so
    host RAM bounds the batch rather than the dataset, and later runs reuse the
    files instead of re-running the encoder.

    Reads go through os.pread rather than a memmap on purpose. Mapping the file
    grows the process RSS as an epoch walks every row, and that memory is
    charged to us; explicit reads leave the caching to the page cache, which
    the kernel can reclaim under pressure.

    See datasets/streams.py for the on-disk layout.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        dataset: TrajectoryDataset,
        cache_dir: "os.PathLike",
        cache_key: str,
        dtype: Any = np.float16,
        embed_goal: bool = False,
        encoder: Optional[EncoderRef] = None,
        dataset_name: str = "",
    ):
        self.dtype = np.dtype(dtype)
        self.root = Path(cache_dir) / cache_key
        encoder = encoder or {"hub_repo": "", "name": "", "feature_key": ""}

        if not StreamStore.exists(self.root):
            if (self.root / "meta.pt").is_file():
                # written before the manifest existed; obs.dat still fits it
                migrate_legacy_cache(
                    self.root, encoder, dataset_name, self.dtype.name
                )
                print(f"########## Migrated embedding cache at {self.root}")
            else:
                self.root.mkdir(parents=True, exist_ok=True)
                self._build(model, dataset, embed_goal, encoder, dataset_name)
                print(f"########## Wrote embedding cache to {self.root}")
        else:
            print(f"########## Reusing embedding cache at {self.root}")

        self.store = StreamStore(self.root)
        self.seq_lengths = self.store.seq_lengths
        assert len(self.seq_lengths) == len(dataset)

    def _build(
        self,
        model: "torch.nn.Module",
        dataset: TrajectoryDataset,
        embed_goal: bool,
        encoder: EncoderRef,
        dataset_name: str,
    ) -> None:
        seq_lengths = [dataset.get_seq_length(i) for i in range(len(dataset))]
        offsets = list(_accumulate([0] + seq_lengths))[:-1]

        accelerator = utils.inference.Accelerator()
        model_device = accelerator.device
        writer = StreamWriter(self.root, encoder, dataset_name)
        writer.disk_stream("obs", "observation", self.dtype.name)
        if embed_goal:
            writer.disk_stream("goal", "goal", self.dtype.name)
        resident: Dict[str, List[torch.Tensor]] = {}

        with utils.inference.eval_mode(model, no_grad=True):
            for i in utils.inference.tqdm(range(len(dataset)), total=len(dataset)):
                sample = dataset[i]
                obs_enc = model(sample["obs"].to(model_device)).detach()
                writer.append(
                    "obs", obs_enc.to("cpu", dtype=torch.float32).numpy()
                )
                for name, value in sample.items():
                    if name == "obs":
                        continue
                    if name == "goal" and embed_goal:
                        goal_enc = model(value.to(model_device)).detach()
                        writer.append(
                            "goal", goal_enc.to("cpu", dtype=torch.float32).numpy()
                        )
                    else:
                        resident.setdefault(name, []).append(value.cpu())

        roles = {"action": "label", "goal": "goal", "state": "proprio"}
        for name, values in resident.items():
            writer.ram_stream(name, roles.get(name, "label"), torch.cat(values, dim=0))
        writer.finalize(seq_lengths, offsets)

    def get_seq_length(self, idx: int) -> int:
        return self.seq_lengths[idx]

    def get_all_actions(self) -> torch.Tensor:
        return self.store.all_rows("action")

    def get_frames(
        self,
        idx: int,
        frames: Sequence[int],
        stream_frames: Optional[StreamFrames] = None,
    ) -> Sample:
        length = self.seq_lengths[idx]

        def local(stream_list: Sequence[int]) -> np.ndarray:
            rows = np.asarray(stream_list, dtype=np.int64)
            # negative indices address the episode, not the row before it
            return np.where(rows < 0, rows + length, rows)

        return {
            name: self.store.read(name, idx, local(frames_for(name, frames, stream_frames)))
            for name in self.store.specs
        }

    def __getitem__(self, idx: int) -> Sample:
        return self.get_frames(idx, range(self.seq_lengths[idx]))

    def __len__(self) -> int:
        return len(self.seq_lengths)


def embedding_cache_key(*parts: Any) -> str:
    """Stable directory name for one (encoder, dataset, dtype) combination."""
    blob = json.dumps([str(p) for p in parts], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def random_split_traj(
    dataset: TrajectoryDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajectorySubset]:
    """
    (Modified from torch.utils.data.dataset.random_split)

    Randomly split a trajectory dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split_traj(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (TrajectoryDataset): TrajectoryDataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        TrajectorySubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set
