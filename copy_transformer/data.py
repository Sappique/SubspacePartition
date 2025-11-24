import torch
import numpy as np


class IterablePureRepeatingPatternDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        num_samples: int,
        vocabulary: list[str],
        context_length: int,
        max_pattern_length: int,
    ):
        """IterableDataset generating sequences with repeated random patterns.

        Args:
            num_samples: Number of samples in the dataset.
            vocabulary: List of tokens to sample from.
            context_length: Length of each sequence.
            max_pattern_length: Maximum length of a random pattern. Can't be larger than the vocabulary size.
            batch_size: Number of samples per batch.
        """

        self._non_iterable_dataset = PureRepeatingPatternDataset(
            num_samples, vocabulary, context_length, max_pattern_length
        )

    def __iter__(self):
        for n in range(len(self._non_iterable_dataset)):
            yield self._non_iterable_dataset[n]


class PureRepeatingPatternDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_samples: int,
        vocabulary: list[str],
        context_length: int,
        max_pattern_length: int,
    ):
        """Dataset generating sequences with repeated random patterns.

        Args:
            num_samples: Number of samples in the dataset.
            vocabulary: List of tokens to sample from.
            context_length: Length of each sequence.
            max_pattern_length: Maximum length of a random pattern. Can't be larger than the vocabulary size.
        """

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
        return 100_000_000

    def __getitem__(self, idx: int) -> str:

        return self.data[idx % len(self.data)]


def random_pattern(vocabulary: list[str], max_length: int) -> str:
    """Generates a random pattern from the given vocabulary containing each token at most once.

    Args:
        vocabulary: List of tokens to sample from.
        max_length: Maximum length of the generated pattern. Can't be larger than the vocabulary size.
    """

    if max_length > len(vocabulary):
        raise ValueError(
            f"max_length ({max_length}) cannot be larger than the vocabulary size ({len(vocabulary)})."
        )

    if max_length < 2:
        raise ValueError("max_length must be at least 2 to form a pattern.")

    length = np.random.randint(2, max_length + 1)

    return "".join(np.random.choice(vocabulary, size=length, replace=False))
