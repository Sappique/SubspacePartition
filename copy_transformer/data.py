import torch
import numpy as np
from abc import abstractmethod
from typing import Any, Sized


class IterableDatasetWrapper(torch.utils.data.IterableDataset):
    """Generic wrapper that makes any Dataset iterable."""

    def __init__(self, dataset: Sized):
        self._dataset = dataset

    def __iter__(self):
        for i in range(len(self._dataset)):
            yield self._dataset[i]  # type: ignore[index]


class InfiniteBufferedDataset(torch.utils.data.IterableDataset):
    """Base class for infinite datasets with sample buffering."""

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.buffer: list[Any] = []
        self.fill_buffer()

    @abstractmethod
    def generate_sample(self) -> Any:
        """Generate a single sample. Override in subclasses."""
        pass

    def fill_buffer(self):
        self.buffer = [self.generate_sample() for _ in range(self.buffer_size)]

    def __iter__(self):
        while True:
            if not self.buffer:
                self.fill_buffer()
            yield self.buffer.pop()


def random_pattern(vocabulary: list[str], max_length: int, min_length: int = 2) -> str:
    """Generates a random pattern from the given vocabulary containing each token at most once.

    Args:
        vocabulary: List of tokens to sample from.
        max_length: Maximum length of the generated pattern. Can't be larger than the vocabulary size.
        min_length: Minimum length of the generated pattern. Must be at least 2.
    """
    if max_length > len(vocabulary):
        raise ValueError(
            f"max_length ({max_length}) cannot be larger than the vocabulary size ({len(vocabulary)})."
        )

    if min_length < 2:
        raise ValueError("min_length must be at least 2 to form a pattern.")

    if min_length > max_length:
        raise ValueError(
            f"min_length ({min_length}) cannot be larger than max_length ({max_length})."
        )

    length = np.random.randint(min_length, max_length + 1)
    return "".join(np.random.choice(vocabulary, size=length, replace=False))


class PureRepeatingPatternDataset(torch.utils.data.Dataset):
    """Dataset generating sequences with repeated random patterns.

    Args:
        num_samples: Number of samples in the dataset.
        vocabulary: List of tokens to sample from.
        context_length: Length of each sequence.
        max_pattern_length: Maximum length of a random pattern. Can't be larger than the vocabulary size.
    """

    def __init__(
        self,
        num_samples: int,
        vocabulary: list[str],
        context_length: int,
        max_pattern_length: int,
    ):
        if max_pattern_length > len(vocabulary):
            raise ValueError(
                f"max_pattern_length ({max_pattern_length}) cannot be larger than the vocabulary size ({len(vocabulary)})."
            )

        self.data: list[str] = []
        for _ in range(num_samples):
            pattern = random_pattern(vocabulary, max_pattern_length)
            repeated_pattern = (pattern * (context_length // len(pattern) + 1))[
                :context_length
            ]
            self.data.append(repeated_pattern)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]


class IterablePureRepeatingPatternDataset(IterableDatasetWrapper):
    """IterableDataset generating sequences with repeated random patterns."""

    def __init__(
        self,
        num_samples: int,
        vocabulary: list[str],
        context_length: int,
        max_pattern_length: int,
    ):
        super().__init__(
            PureRepeatingPatternDataset(
                num_samples, vocabulary, context_length, max_pattern_length
            )
        )


class InfinitePureRepeatingPatternDataset(InfiniteBufferedDataset):
    """IterableDataset generating infinite sequences with repeated random patterns."""

    def __init__(
        self,
        vocabulary: list[str],
        context_length: int,
        max_pattern_length: int,
        buffer_size: int = 1000,
    ):
        self.vocabulary = vocabulary
        self.context_length = context_length
        self.max_pattern_length = max_pattern_length
        super().__init__(buffer_size)

    def generate_sample(self) -> str:
        pattern = random_pattern(self.vocabulary, self.max_pattern_length)
        return (pattern * (self.context_length // len(pattern) + 1))[
            : self.context_length
        ]


class UniqueTokenPatternDataset(torch.utils.data.Dataset):
    """Dataset generating sequences where a unique token pattern repeats exactly once.

    Each pattern consists of unique tokens (sampled without replacement) and is
    repeated exactly once to form sequences like "ABCABC" or "2JI82JI8".

    Args:
        num_samples: Number of samples in the dataset.
        vocabulary: List of tokens to sample from.
        min_pattern_length: Minimum length of a random pattern. Must be at least 2.
        max_pattern_length: Maximum length of a random pattern. Can't be larger than vocabulary size.
        mask_first_repetition: If True, returns dict with "text" and "mask" keys where mask
            is 0 for the first pattern occurrence and 1 for the second. If False, returns
            plain strings.
    """

    def __init__(
        self,
        num_samples: int,
        vocabulary: list[str],
        min_pattern_length: int,
        max_pattern_length: int,
        mask_first_repetition: bool = False,
    ):
        if max_pattern_length > len(vocabulary):
            raise ValueError(
                f"max_pattern_length ({max_pattern_length}) cannot be larger than "
                f"the vocabulary size ({len(vocabulary)})."
            )

        self.mask_first_repetition = mask_first_repetition
        self.data: list[dict[str, Any] | str] = []

        for _ in range(num_samples):
            pattern = random_pattern(vocabulary, max_pattern_length, min_pattern_length)
            sequence = pattern * 2  # Repeat exactly once

            if mask_first_repetition:
                # Mask: 0 for first occurrence, 1 for second
                mask = [0] * len(pattern) + [1] * len(pattern)
                self.data.append({"text": sequence, "mask": mask})
            else:
                self.data.append(sequence)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any] | str:
        return self.data[idx]


class IterableUniqueTokenPatternDataset(IterableDatasetWrapper):
    """IterableDataset generating sequences where a unique token pattern repeats exactly once."""

    def __init__(
        self,
        num_samples: int,
        vocabulary: list[str],
        min_pattern_length: int,
        max_pattern_length: int,
        mask_first_repetition: bool = False,
    ):
        super().__init__(
            UniqueTokenPatternDataset(
                num_samples,
                vocabulary,
                min_pattern_length,
                max_pattern_length,
                mask_first_repetition,
            )
        )


class InfiniteUniqueTokenPatternDataset(InfiniteBufferedDataset):
    """IterableDataset generating infinite sequences where a unique token pattern repeats exactly once."""

    def __init__(
        self,
        vocabulary: list[str],
        min_pattern_length: int,
        max_pattern_length: int,
        buffer_size: int = 1000,
        mask_first_repetition: bool = False,
    ):
        if max_pattern_length > len(vocabulary):
            raise ValueError(
                f"max_pattern_length ({max_pattern_length}) cannot be larger than "
                f"the vocabulary size ({len(vocabulary)})."
            )

        self.vocabulary = vocabulary
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.mask_first_repetition = mask_first_repetition
        super().__init__(buffer_size)

    def generate_sample(self) -> dict[str, Any] | str:
        pattern = random_pattern(
            self.vocabulary, self.max_pattern_length, self.min_pattern_length
        )
        sequence = pattern * 2
        if self.mask_first_repetition:
            return {"text": sequence, "mask": [0] * len(pattern) + [1] * len(pattern)}
        return sequence
