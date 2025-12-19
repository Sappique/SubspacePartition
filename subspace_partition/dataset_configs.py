"""Dataset configuration classes following best practices for serialization.

This module uses dataclasses with to_dict/from_dict methods for clean serialization.
Extensible design allows adding new dataset types easily.
"""

import torch
from dataclasses import dataclass
from torch.utils.data import IterableDataset, Dataset
from typing import Union, Any, Literal
from abc import ABC, abstractmethod


@dataclass
class DatasetConfig(ABC):
    """Abstract base class for all dataset configurations.

    To add a new dataset type:
    1. Create a new dataclass inheriting from DatasetConfig
    2. Implement to_dict(), from_dict(), and create_dataset()
    3. Add to DATASET_CONFIG_REGISTRY with a unique type string
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. Must include a 'type' key."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetConfig":
        """Deserialize config from a dictionary."""
        pass

    @abstractmethod
    def create_dataset(self) -> Union[IterableDataset, Dataset]:
        """Create the actual dataset instance from this config."""
        pass


@dataclass
class PureRepeatingPatternConfig(DatasetConfig):
    """Unified config for all PureRepeatingPattern dataset variants.

    Args:
        vocabulary: List of tokens to sample from.
        context_length: Length of each sequence.
        max_pattern_length: Maximum length of a random pattern.
        iterable: If True, creates an IterableDataset. If False, creates a regular Dataset.
        length: Number of samples. Can be an integer or the string "infinite" for infinite datasets.
        buffer_size: Buffer size for infinite datasets (only used when length="infinite").

    Examples:
        >>> # Finite iterable dataset with 1000 samples
        >>> config = PureRepeatingPatternConfig(
        ...     vocabulary=["a", "b", "c"],
        ...     context_length=64,
        ...     max_pattern_length=4,
        ...     iterable=True,
        ...     length=1000
        ... )

        >>> # Infinite iterable dataset
        >>> config = PureRepeatingPatternConfig(
        ...     vocabulary=["a", "b", "c"],
        ...     context_length=64,
        ...     max_pattern_length=4,
        ...     iterable=True,
        ...     length="infinite",
        ...     buffer_size=1000
        ... )

        >>> # Finite non-iterable dataset
        >>> config = PureRepeatingPatternConfig(
        ...     vocabulary=["a", "b", "c"],
        ...     context_length=64,
        ...     max_pattern_length=4,
        ...     iterable=False,
        ...     length=500
        ... )
    """

    vocabulary: list[str]
    context_length: int
    max_pattern_length: int
    iterable: bool = True
    length: int | Literal["infinite"] = "infinite"
    buffer_size: int = 1000

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary."""
        return {
            "type": "pure_repeating_pattern",  # Type identifier for registry
            "vocabulary": self.vocabulary,
            "context_length": self.context_length,
            "max_pattern_length": self.max_pattern_length,
            "iterable": self.iterable,
            "length": self.length,
            "buffer_size": self.buffer_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PureRepeatingPatternConfig":
        """Deserialize config from a dictionary."""
        # Remove 'type' key if present
        config_data = {k: v for k, v in data.items() if k != "type"}
        return cls(
            vocabulary=config_data["vocabulary"],
            context_length=config_data["context_length"],
            max_pattern_length=config_data["max_pattern_length"],
            iterable=config_data.get("iterable", True),
            length=config_data.get("length", "infinite"),
            buffer_size=config_data.get("buffer_size", 1000),
        )

    def create_dataset(self) -> Union[IterableDataset, Dataset]:
        """Create the actual dataset instance from this config."""
        if self.length == "infinite":
            if not self.iterable:
                raise ValueError("Non-iterable datasets cannot have infinite length")
            from copy_transformer.data import InfinitePureRepeatingPatternDataset

            return InfinitePureRepeatingPatternDataset(
                vocabulary=self.vocabulary,
                context_length=self.context_length,
                max_pattern_length=self.max_pattern_length,
                buffer_size=self.buffer_size,
            )
        else:
            # Finite length
            if not isinstance(self.length, int):
                raise ValueError(
                    f"length must be an integer or 'infinite', got {self.length}"
                )

            if self.iterable:
                from copy_transformer.data import IterablePureRepeatingPatternDataset

                return IterablePureRepeatingPatternDataset(
                    num_samples=self.length,
                    vocabulary=self.vocabulary,
                    context_length=self.context_length,
                    max_pattern_length=self.max_pattern_length,
                )
            else:
                from copy_transformer.data import PureRepeatingPatternDataset

                return PureRepeatingPatternDataset(
                    num_samples=self.length,
                    vocabulary=self.vocabulary,
                    context_length=self.context_length,
                    max_pattern_length=self.max_pattern_length,
                )


# Registry mapping type strings to config classes
DATASET_CONFIG_REGISTRY: dict[str, type[DatasetConfig]] = {
    "pure_repeating_pattern": PureRepeatingPatternConfig,
}


def register_dataset_config(type_name: str, config_class: type[DatasetConfig]):
    """Register a new dataset config type.

    Args:
        type_name: Unique string identifier for this dataset type.
        config_class: DatasetConfig subclass to register.

    Example:
        >>> register_dataset_config("my_custom_dataset", MyCustomDatasetConfig)
    """
    if type_name in DATASET_CONFIG_REGISTRY:
        raise ValueError(f"Dataset type '{type_name}' is already registered")
    DATASET_CONFIG_REGISTRY[type_name] = config_class


def dataset_config_from_dict(data: dict[str, Any]) -> DatasetConfig:
    """Deserialize a dataset config from a dictionary using the registry.

    Args:
        data: Dictionary with dataset configuration parameters.
              Must include a 'type' key matching a registered config type.

    Returns:
        Appropriate DatasetConfig subclass instance.

    Raises:
        ValueError: If 'type' key is missing or not registered.

    Example:
        >>> config_dict = {
        ...     "type": "pure_repeating_pattern",
        ...     "vocabulary": ["a", "b"],
        ...     "context_length": 64,
        ...     "max_pattern_length": 4,
        ...     "iterable": True,
        ...     "length": "infinite",
        ... }
        >>> config = dataset_config_from_dict(config_dict)
        >>> dataset = config.create_dataset()
    """
    dataset_type = data.get("type")

    if dataset_type is None:
        raise ValueError("Dataset config dictionary must include a 'type' key")

    if dataset_type not in DATASET_CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown dataset type: '{dataset_type}'. "
            f"Registered types: {list(DATASET_CONFIG_REGISTRY.keys())}"
        )

    config_class = DATASET_CONFIG_REGISTRY[dataset_type]
    return config_class.from_dict(data)


# Backward compatibility: keep get_dataset for convenience
def get_dataset(
    vocabulary: list[str],
    context_length: int,
    max_pattern_length: int,
    iterable: bool = True,
    length: Union[int, Literal["infinite"]] = "infinite",
    buffer_size: int = 1000,
    **kwargs,
) -> Union[IterableDataset, Dataset]:
    """Factory function to create datasets (backward compatible).

    Args:
        vocabulary: List of tokens to sample from.
        context_length: Length of each sequence.
        max_pattern_length: Maximum length of a random pattern.
        iterable: If True, creates an IterableDataset.
        length: Number of samples or "infinite".
        buffer_size: Buffer size for infinite datasets.
        **kwargs: Additional arguments (for backward compatibility).

    Returns:
        Dataset instance.
    """
    config = PureRepeatingPatternConfig(
        vocabulary=vocabulary,
        context_length=context_length,
        max_pattern_length=max_pattern_length,
        iterable=iterable,
        length=length,
        buffer_size=buffer_size,
    )
    return config.create_dataset()
