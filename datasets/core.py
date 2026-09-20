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

from datasets.types import Sample


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
    def get_frames(self, idx: int, frames: Sequence[int]) -> Sample:
        """
        Returns the frames from the idx-th trajectory at the specified frames.
        Used to speed up slicing.
        """
        raise NotImplementedError


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

    def get_frames(self, idx: int, frames: Sequence[int]) -> Sample:
        return self.dataset.get_frames(self.indices[idx], frames)


class TrajectorySlicerDataset(Dataset):
    """
    Slice a trajectory dataset into (overlapping) windows of `window` observations
    paired with an action chunk.

    dataset: a trajectory dataset that satisfies:
        dataset.get_seq_length(i) returns the length of sequence i
        dataset.get_frames(i, frames) -> Sample, each stream sliced to frames
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
        i, start, end = self.slices[idx]
        # fetch only the frames this slice needs (the union of the obs window and
        # the action window, which can extend past it) instead of the whole
        # episode: dataset[i] reprocesses every frame of the episode, while
        # get_frames does work proportional to what's requested
        T = self.dataset.get_seq_length(i)
        # -1 due to overlap for 1 step between obs and act
        frame_end = min(end - 1 + self.action_window, T)
        sample = self.dataset.get_frames(i, range(start, frame_end))
        # "action" spans the action window; every other stream follows the
        # observation window, which is the shorter of the two
        act = sample["action"]
        rel_end = end - start

        values: Sample = {}
        if end - start < self.window:
            act = utils.inference.repeat_start_to_length(
                act, self.window + self.action_window - 1, dim=0
            )
            for name, value in sample.items():
                if name == "action":
                    continue
                values[name] = utils.inference.repeat_start_to_length(
                    value[:rel_end], self.window, dim=0
                )
        else:
            for name, value in sample.items():
                if name == "action":
                    continue
                values[name] = value[:rel_end]

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

    def get_frames(self, idx: int, frames: Sequence[int]) -> Sample:
        return {name: value[frames] for name, value in self.data[idx].items()}

    def __getitem__(self, idx: int) -> Sample:
        # the stored tensors, not get_frames(range(...)): a range index copies the episode
        return dict(self.data[idx])

    def __len__(self) -> int:
        return len(self.seq_lengths)


class FileEmbeddingDataset(TrajectoryDataset):
    """Precomputed patch embeddings kept in a file instead of in RAM.

    TrajectoryEmbeddingDataset holds every episode's features resident, which
    is ~30 GiB for Cube at fp32 and does not fit a 31 GB host. This writes them
    once into a flat file and reads back only the rows a sample needs, so host
    RAM bounds the batch rather than the dataset, and the file is reused by
    later runs instead of re-running the encoder.

    Reads go through os.pread rather than a memmap on purpose. Mapping the file
    grows the process RSS as an epoch touches every row, and that memory is
    charged to us; explicit reads leave the caching to the page cache, which
    the kernel can reclaim under pressure.

    The actions and goals stay in RAM. They are a few MB, and the goal is a
    dummy tensor for every environment except LIBERO.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        dataset: TrajectoryDataset,
        cache_dir: "os.PathLike",
        cache_key: str,
        dtype: Any = np.float16,
        embed_goal: bool = False,
    ):
        self.dtype = np.dtype(dtype)
        cache_dir = Path(cache_dir) / cache_key
        self.obs_path = cache_dir / "obs.dat"
        self.meta_path = cache_dir / "meta.pt"

        if not (self.obs_path.exists() and self.meta_path.exists()):
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._build(model, dataset, embed_goal)
            print(f"########## Wrote embedding cache to {cache_dir}")
        else:
            print(f"########## Reusing embedding cache at {cache_dir}")

        meta = torch.load(self.meta_path, weights_only=False)
        self.seq_lengths = meta["seq_lengths"]
        self.offsets = meta["offsets"]
        self.rest: List[Sample] = [_as_sample(r) for r in meta["rest"]]
        self.frame_shape = tuple(meta["frame_shape"])
        assert len(self.seq_lengths) == len(dataset)

        # bytes per frame: the row stride in obs.dat
        self.row_stride = int(np.prod(self.frame_shape)) * self.dtype.itemsize
        self.num_rows = sum(self.seq_lengths)
        # opened lazily, and re-opened per process: dataloader workers are
        # forked, and a descriptor is cheaper to remake than to reason about
        self._fd: Optional[int] = None
        self._fd_pid: Optional[int] = None

    def _build(
        self,
        model: "torch.nn.Module",
        dataset: TrajectoryDataset,
        embed_goal: bool,
    ) -> None:
        seq_lengths = [dataset.get_seq_length(i) for i in range(len(dataset))]
        offsets = list(_accumulate([0] + seq_lengths))[:-1]

        accelerator = utils.inference.Accelerator()
        model_device = accelerator.device
        frame_shape: Optional[Tuple[int, ...]] = None
        rest_all: List[Sample] = []

        # written straight through: the pass is sequential, so a plain handle
        # avoids the dirty-page build-up a write-mode memmap accumulates
        with open(self.obs_path, "wb") as fh:
            with utils.inference.eval_mode(model, no_grad=True):
                for i in utils.inference.tqdm(range(len(dataset)), total=len(dataset)):
                    sample = dataset[i]
                    obs_enc = model(sample["obs"].to(model_device)).detach()
                    if frame_shape is None:
                        # only known after the first forward pass
                        frame_shape = tuple(obs_enc.shape[1:])
                    fh.write(
                        obs_enc.to("cpu", dtype=torch.float32)
                        .numpy()
                        .astype(self.dtype)
                        .tobytes()
                    )
                    rest: Sample = {
                        name: value.cpu()
                        for name, value in sample.items()
                        if name != "obs"
                    }
                    if embed_goal:
                        rest["goal"] = (
                            model(sample["goal"].to(model_device)).detach().cpu()
                        )
                    rest_all.append(rest)

        torch.save(
            {
                "seq_lengths": seq_lengths,
                "offsets": offsets,
                "rest": rest_all,
                "frame_shape": frame_shape,
            },
            self.meta_path,
        )

    def _descriptor(self) -> int:
        pid = os.getpid()
        if self._fd is None or self._fd_pid != pid:
            self._fd = os.open(str(self.obs_path), os.O_RDONLY)
            self._fd_pid = pid
        return self._fd

    def _read_rows(self, rows: np.ndarray) -> torch.Tensor:
        """Read whole frames by row index, coalescing consecutive rows."""
        fd = self._descriptor()
        chunks: List[np.ndarray] = []
        start = 0
        while start < len(rows):
            # consecutive rows are contiguous on disk, so one read serves them
            end = start + 1
            while end < len(rows) and rows[end] == rows[end - 1] + 1:
                end += 1
            count = end - start
            raw = os.pread(
                fd, count * self.row_stride, int(rows[start]) * self.row_stride
            )
            chunks.append(np.frombuffer(raw, dtype=self.dtype))
            start = end
        flat = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        # np.frombuffer is read-only; torch needs a writable buffer to own
        return torch.from_numpy(flat.reshape(len(rows), *self.frame_shape).copy())

    def get_seq_length(self, idx: int) -> int:
        return self.seq_lengths[idx]

    def get_all_actions(self) -> torch.Tensor:
        return torch.cat([rest["action"] for rest in self.rest], dim=0)

    def get_frames(self, idx: int, frames: Sequence[int]) -> Sample:
        length = self.seq_lengths[idx]
        local = np.asarray(frames, dtype=np.int64)
        # negative indices address the episode, not the row before it
        local = np.where(local < 0, local + length, local)
        sample: Sample = {"obs": self._read_rows(local + self.offsets[idx])}
        for name, value in self.rest[idx].items():
            sample[name] = value[local]
        return sample

    def __getitem__(self, idx: int) -> Sample:
        return self.get_frames(idx, range(self.seq_lengths[idx]))

    def __len__(self) -> int:
        return len(self.seq_lengths)


def _as_sample(rest: Any) -> Sample:
    """Name the non-observation tensors of a cache written before Sample existed."""
    if isinstance(rest, dict):
        return rest
    # positional order was (action, goal), per "assuming goal comes last"
    names = ("action", "goal")
    return {names[i]: value for i, value in enumerate(rest)}


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
