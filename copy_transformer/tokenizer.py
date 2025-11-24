from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encodes the given text into a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into a string."""
        pass


class SingleCharTokenizer(BaseTokenizer):
    """A extremely simple tokenizer for single-character tokens."""

    def __init__(self, vocabulary: list[str]):
        """Initializes the tokenizer with the given vocabulary.

        Args:
            vocabulary: List of single-character tokens.
        """

        if any(len(char) != 1 for char in vocabulary):
            raise ValueError("All tokens in the vocabulary must be single characters.")

        self.char_to_id = {char: idx for idx, char in enumerate(vocabulary)}
        self.id_to_char = {idx: char for idx, char in enumerate(vocabulary)}

    def encode(self, text: str) -> list[int]:
        """Encodes the given text into a list of token IDs.

        Args:
            text: The input text to encode.

        Returns:
            A list of token IDs corresponding to the input text.

        Raises:
            ValueError: If a character in the text is not in the vocabulary.
        """
        try:
            return [self.char_to_id[char] for char in text]
        except KeyError as e:
            raise ValueError(f"Character '{e.args[0]}' not in vocabulary.")

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into a string.

        Args:
            ids: A list of token IDs to decode.

        Returns:
            The decoded string.

        Raises:
            ValueError: If a token ID is not in the vocabulary.
        """
        try:
            return "".join([self.id_to_char[id] for id in ids])
        except KeyError as e:
            raise ValueError(f"ID '{e.args[0]}' not in vocabulary.")
