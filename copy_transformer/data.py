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


def _hierholzer_eulerian_circuit(
    adj: dict[str, list[str]], start_node: str
) -> list[str]:
    """Find an Eulerian circuit using Hierholzer's algorithm.

    Args:
        adj: Adjacency list mapping each node to its outgoing edges (modified in place).
        start_node: Starting node for the circuit.

    Returns:
        List of nodes forming the Eulerian circuit.
    """
    stack = [start_node]
    circuit = []

    while stack:
        v = stack[-1]
        if adj[v]:
            stack.append(adj[v].pop())
        else:
            circuit.append(stack.pop())

    circuit.reverse()
    return circuit


def _generate_all_ngram_prefixes(vocab: list[str], n: int) -> list[str]:
    """Generate all (n-1)-grams as nodes for the de Bruijn graph."""
    if n == 1:
        return [""]  # Single empty prefix for unigrams

    from itertools import product

    return ["".join(p) for p in product(vocab, repeat=n - 1)]


def random_ngram_pattern(
    vocabulary: list[str],
    n: int,
    max_length: int,
    min_length: int = 2,
    use_only_n_unique_tokens: int | None = None,
) -> str:
    """Generates a random pattern where all n-grams are unique.

    All n-grams of size n within the pattern are guaranteed to be unique.
    Tokens within n-grams are sampled with replacement (tokens can repeat),
    but n-grams themselves must be unique.

    Uses Hierholzer's algorithm to find an Eulerian circuit in the de Bruijn
    graph, guaranteeing O(N^n) time complexity and success whenever a solution
    exists (when M - (n-1) <= N^n, where M is pattern length and N is vocab size).

    Args:
        vocabulary: List of tokens to sample from.
        n: Size of n-grams to ensure uniqueness for.
        max_length: Maximum length of the generated pattern in tokens.
        min_length: Minimum length of the generated pattern in tokens. Must be at least n.
        use_only_n_unique_tokens: If set, only use this many unique tokens per pattern.

    Raises:
        ValueError: If parameters are invalid or no valid pattern exists.
    """
    if n < 1:
        raise ValueError(f"n ({n}) must be at least 1.")

    if min_length < n:
        raise ValueError(f"min_length ({min_length}) must be at least n ({n}).")

    if min_length > max_length:
        raise ValueError(
            f"min_length ({min_length}) cannot be larger than max_length ({max_length})."
        )

    if use_only_n_unique_tokens is not None:
        if use_only_n_unique_tokens > len(vocabulary):
            raise ValueError(
                f"use_only_n_unique_tokens ({use_only_n_unique_tokens}) cannot be larger "
                f"than the vocabulary size ({len(vocabulary)})."
            )
        if use_only_n_unique_tokens < 1:
            raise ValueError("use_only_n_unique_tokens must be at least 1.")
        vocab = list(
            np.random.choice(vocabulary, size=use_only_n_unique_tokens, replace=False)
        )
    else:
        vocab = list(vocabulary)

    target_length = np.random.randint(min_length, max_length + 1)
    N = len(vocab)

    # Number of n-grams needed: target_length - (n - 1)
    # Maximum possible n-grams: N^n
    num_ngrams_needed = target_length - (n - 1)
    max_ngrams = N**n

    if num_ngrams_needed > max_ngrams:
        raise ValueError(
            f"Cannot generate unique {n}-gram pattern of length {target_length}: "
            f"need {num_ngrams_needed} unique n-grams but only {max_ngrams} exist "
            f"with vocabulary of size {N}."
        )

    if target_length == 0:
        return ""
    if target_length < n:
        # Not enough tokens to form any n-gram, just return random tokens
        return "".join(vocab[np.random.randint(N)] for _ in range(target_length))

    # Build de Bruijn graph:
    # - Nodes are (n-1)-grams
    # - Edge from prefix to suffix represents an n-gram
    prefixes = _generate_all_ngram_prefixes(vocab, n)

    # Build adjacency lists (shuffled for randomness)
    # Each (n-1)-gram prefix can be extended by any character
    adj: dict[str, list[str]] = {}
    for prefix in prefixes:
        # Edges go from prefix to (prefix[1:] + char) for each char
        # But we track by storing the next node (suffix)
        extensions = []
        for char in vocab:
            if n == 1:
                suffix = ""
            else:
                suffix = prefix[1:] + char if len(prefix) > 0 else char
            extensions.append(suffix)
        np.random.shuffle(extensions)
        adj[prefix] = extensions

    # Pick random starting node
    start = prefixes[np.random.randint(len(prefixes))]

    # Find Eulerian circuit (visits all N^n edges)
    circuit = _hierholzer_eulerian_circuit(adj, start)

    # Circuit has N^n + 1 nodes; extract substring of target_length
    # The circuit represents: each consecutive pair of nodes shares an edge (n-gram)
    # To get the actual string: first node gives first (n-1) chars, then each
    # subsequent node contributes its last character

    if n == 1:
        # For unigrams, circuit nodes are empty strings; we need to track differently
        # Rebuild: just sample without replacement
        result_chars = list(vocab)
        np.random.shuffle(result_chars)
        return "".join(result_chars[:target_length])

    # For n >= 2: reconstruct string from circuit
    # First node is an (n-1)-gram, subsequent nodes each add one character
    full_string_chars = list(circuit[0])  # First (n-1) characters
    for i in range(1, len(circuit)):
        # Each subsequent node's last character is the new character
        full_string_chars.append(circuit[i][-1])

    # Pick random starting position
    full_length = len(full_string_chars)
    if full_length < target_length:
        # Shouldn't happen if math is right, but safety check
        raise ValueError(
            f"Internal error: Eulerian circuit produced {full_length} chars, "
            f"need {target_length}."
        )

    start_pos = np.random.randint(0, full_length - target_length + 1)
    return "".join(full_string_chars[start_pos : start_pos + target_length])


class UniqueNgramPatternDataset(torch.utils.data.Dataset):
    """Dataset generating sequences where a unique n-gram pattern repeats exactly once.

    Each pattern consists of tokens where all n-grams (of size n) are unique.
    Individual tokens may repeat, but n-grams must be unique. A separator token
    (e.g., BOS) is inserted between the two repetitions to prevent boundary n-gram
    issues.

    Args:
        num_samples: Number of samples in the dataset.
        vocabulary: List of tokens to sample from.
        n: Size of n-grams to ensure uniqueness for.
        min_pattern_length: Minimum length of pattern in tokens. Must be at least n.
        max_pattern_length: Maximum length of pattern in tokens.
        separator: Token to insert between the two pattern repetitions. Should not
            be in the vocabulary to avoid confusion.
        mask_first_repetition: If True, returns dict with "text" and "mask" keys where mask
            is 0 for the first pattern occurrence (and separator) and 1 for the second.
            If False, returns plain strings.
        use_only_n_unique_tokens_per_pattern: If set, only use this many unique tokens
            from vocabulary for each pattern (sampled randomly per pattern).
    """

    def __init__(
        self,
        num_samples: int,
        vocabulary: list[str],
        n: int,
        min_pattern_length: int,
        max_pattern_length: int,
        separator: str,
        mask_first_repetition: bool = False,
        use_only_n_unique_tokens_per_pattern: int | None = None,
    ):
        if use_only_n_unique_tokens_per_pattern is not None:
            if use_only_n_unique_tokens_per_pattern > len(vocabulary):
                raise ValueError(
                    f"use_only_n_unique_tokens_per_pattern ({use_only_n_unique_tokens_per_pattern}) "
                    f"cannot be larger than the vocabulary size ({len(vocabulary)})."
                )

        self.separator = separator
        self.mask_first_repetition = mask_first_repetition
        self.data: list[dict[str, Any] | str] = []

        for _ in range(num_samples):
            pattern = random_ngram_pattern(
                vocabulary,
                n,
                max_pattern_length,
                min_pattern_length,
                use_only_n_unique_tokens=use_only_n_unique_tokens_per_pattern,
            )
            # Insert separator between repetitions: pattern + separator + pattern
            sequence = pattern + separator + pattern

            if mask_first_repetition:
                # Mask: 0 for first occurrence + separator, 1 for second occurrence
                mask = [0] * (len(pattern) + 1) + [1] * len(pattern)
                self.data.append({"text": sequence, "mask": mask})
            else:
                self.data.append(sequence)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any] | str:
        return self.data[idx]


class IterableUniqueNgramPatternDataset(IterableDatasetWrapper):
    """IterableDataset generating sequences where a unique n-gram pattern repeats exactly once."""

    def __init__(
        self,
        num_samples: int,
        vocabulary: list[str],
        n: int,
        min_pattern_length: int,
        max_pattern_length: int,
        separator: str,
        mask_first_repetition: bool = False,
        use_only_n_unique_tokens_per_pattern: int | None = None,
    ):
        super().__init__(
            UniqueNgramPatternDataset(
                num_samples,
                vocabulary,
                n,
                min_pattern_length,
                max_pattern_length,
                separator,
                mask_first_repetition,
                use_only_n_unique_tokens_per_pattern,
            )
        )


class InfiniteUniqueNgramPatternDataset(InfiniteBufferedDataset):
    """IterableDataset generating infinite sequences where a unique n-gram pattern repeats exactly once."""

    def __init__(
        self,
        vocabulary: list[str],
        n: int,
        min_pattern_length: int,
        max_pattern_length: int,
        separator: str,
        buffer_size: int = 1000,
        mask_first_repetition: bool = False,
        use_only_n_unique_tokens_per_pattern: int | None = None,
    ):
        if use_only_n_unique_tokens_per_pattern is not None:
            if use_only_n_unique_tokens_per_pattern > len(vocabulary):
                raise ValueError(
                    f"use_only_n_unique_tokens_per_pattern ({use_only_n_unique_tokens_per_pattern}) "
                    f"cannot be larger than the vocabulary size ({len(vocabulary)})."
                )

        self.vocabulary = vocabulary
        self.n = n
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.separator = separator
        self.mask_first_repetition = mask_first_repetition
        self.use_only_n_unique_tokens_per_pattern = use_only_n_unique_tokens_per_pattern
        super().__init__(buffer_size)

    def generate_sample(self) -> dict[str, Any] | str:
        pattern = random_ngram_pattern(
            self.vocabulary,
            self.n,
            self.max_pattern_length,
            self.min_pattern_length,
            use_only_n_unique_tokens=self.use_only_n_unique_tokens_per_pattern,
        )
        # Insert separator between repetitions: pattern + separator + pattern
        sequence = pattern + self.separator + pattern
        if self.mask_first_repetition:
            return {
                "text": sequence,
                "mask": [0] * (len(pattern) + 1) + [1] * len(pattern),
            }
        return sequence
