from pathlib import Path
from transformers import PreTrainedTokenizer
from typing import List, Optional


class SingleCharTokenizer(PreTrainedTokenizer):
    """Super simple character tokenizer that takes a list of chars as alphabet."""

    def __init__(self, alphabet: List[str], **kwargs):
        """Creates a character-level tokenizer.

        Args:
            alphabet: List of characters to use as vocabulary. Special tokens
                should not be included in this list; they can be specified via
                the kwargs `bos_token`, `eos_token`, `unk_token`, and
                `pad_token`.
        """

        if any(len(char) != 1 for char in alphabet):
            raise ValueError("All elements in alphabet must be single characters.")

        self.alphabet = alphabet

        self.char_to_id = {char: idx for idx, char in enumerate(alphabet)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}

        super().__init__(**kwargs)

        # Add special tokens to vocab after parent init
        for token, token_id in [
            (self.bos_token, self.bos_token_id),
            (self.eos_token, self.eos_token_id),
            (self.unk_token, self.unk_token_id),
            (self.pad_token, self.pad_token_id),
        ]:
            if token is not None and token_id is not None:
                self.char_to_id[token] = token_id
                self.id_to_char[token_id] = token

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size."""

        return len(self.char_to_id)

    def get_vocab(self):
        """Returns the vocabulary as a dictionary mapping characters to IDs."""

        return self.char_to_id.copy()

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Splits the text into individual characters."""

        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token to its ID.

        If the token is not found the ID of the unk_token is returned, if the
        unk_token is set. Otherwise, a KeyError is raised.
        """

        try:
            return self.char_to_id[token]
        except KeyError:
            if self.unk_token is not None:
                return self.char_to_id[self.unk_token]
            else:
                raise KeyError(
                    f"Token '{token}' not in vocabulary and unk_token is not set."
                )

    def _convert_id_to_token(self, index: int) -> str:
        """Converts a ID to a token."""

        return self.id_to_char[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Join tokens into a string."""

        return "".join(tokens)
