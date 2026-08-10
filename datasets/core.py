import abc
import utils
import torch
import numpy as np
from torch import default_generator, randperm
from torch.utils.data import Dataset, Subset
from typing import Callable, Optional, Sequence, List
from torch.nn.utils.rnn import pad_sequence


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
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: False: invalid; True: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_frames(self, idx, frames):
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

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def get_all_actions(self):
        return self.dataset.get_all_actions()

    def get_frames(self, idx, frames):
        return self.dataset.get_frames(self.indices[idx], frames)


class TrajectorySlicerDataset(Dataset):
    """
    Slice a trajectory dataset into (overlapping) windows of `window` observations
    paired with an action chunk.

    dataset: a trajectory dataset that satisfies:
        dataset.get_seq_length(i) returns the length of sequence i
        dataset[i] = (observations, actions, *others)
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

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        obs, act, *others = self.dataset[i]
        if end - start < self.window:
            obs_win = utils.inference.repeat_start_to_length(
                obs[start:end], self.window, dim=0
            )
            act = utils.inference.repeat_start_to_length(
                # -1 due to overlap for 1 step between obs and act
                act[start : end - 1 + self.action_window],
                self.window + self.action_window - 1,
                dim=0,
            )
            repeated_others = [
                utils.inference.repeat_start_to_length(
                    other[start:end], self.window, dim=0
                )
                for other in others
            ]
        else:
            obs_win = obs[start:end]
            # -1 due to overlap for 1 step between obs and act
            act = act[start : end - 1 + self.action_window]
            repeated_others = [other[start:end] for other in others]

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
        values = [obs_win, act, *repeated_others]

        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        return tuple(values)


class TrajectoryEmbeddingDataset(TrajectoryDataset):
    def __init__(
        self,
        model,
        dataset: TrajectoryDataset,
        device="cpu",
        embed_goal=False,
    ):
        self.data = utils.inference.embed_trajectory_dataset(
            model,
            dataset,
            obs_only=False,
            device=device,
            embed_goal=embed_goal,
        )
        assert len(self.data) == len(dataset)

        self.seq_lengths = [len(x[0]) for x in self.data]
        self.on_device_data = []
        n_tensors = len(self.data[0])
        for i in range(n_tensors):
            self.on_device_data.append(
                pad_sequence([x[i] for x in self.data], batch_first=True).to(device)
            )
        self.data = self.on_device_data

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return torch.cat([x[1] for x in self.data], dim=0)

    def get_frames(self, idx, frames):
        return [x[idx, frames] for x in self.data]

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


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
