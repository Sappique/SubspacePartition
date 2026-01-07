import json
from pathlib import Path
from typing import Literal, Any
from dataclasses import dataclass, field
import torch
import subspace_partition.serialization

from transformers import PreTrainedTokenizerBase
from transformer_lens import HookedTransformerConfig
from subspace_partition.dataset_configs import DatasetConfig


@dataclass
class TrainingConfig:
    """Configuration for transformer training.

    This class handles serialization of training parameters while keeping
    model_config and training_args in separate files.

    Args:
        model_name: Name for saving/loading the model.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        batch_size: Batch size for training.
        validate: Whether to run validation.
        validation_batch_size: Batch size for validation (defaults to batch_size).
        optimizer_type: Type of optimizer ('adam', etc.).
        loss_fn_type: Type of loss function ('cross_entropy', etc.).
        dataset_config: Configuration for the training dataset (required).
        validation_dataset_config: Optional configuration for validation dataset.
    """

    model_name: str
    epochs: int
    dataset_config: DatasetConfig
    learning_rate: float = 1e-3
    batch_size: int = 32
    validate: bool = True
    validation_batch_size: int | None = None
    optimizer_type: str = "adam"
    loss_fn_type: str = "cross_entropy"
    validation_dataset_config: DatasetConfig | None = None

    def __post_init__(self):
        if self.validation_batch_size is None:
            self.validation_batch_size = self.batch_size

    def to_dict(self) -> dict[str, Any]:
        """Serialize training config to dictionary."""
        config_dict = {
            "model_name": self.model_name,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "validate": self.validate,
            "validation_batch_size": self.validation_batch_size,
            "optimizer_type": self.optimizer_type,
            "loss_fn_type": self.loss_fn_type,
        }

        # Serialize dataset configs
        config_dict["dataset_config"] = self.dataset_config.to_dict()

        if self.validation_dataset_config is not None:
            config_dict["validation_dataset_config"] = (
                self.validation_dataset_config.to_dict()
            )

        return config_dict

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        dataset_config: DatasetConfig | None = None,
        validation_dataset_config: DatasetConfig | None = None,
    ) -> "TrainingConfig":
        """Deserialize training config from dictionary.

        Args:
            data: Dictionary with training configuration.
            dataset_config: Optional dataset config (overrides one in dict).
            validation_dataset_config: Optional validation dataset config.
        """
        from subspace_partition.dataset_configs import dataset_config_from_dict

        # Reconstruct dataset configs
        if dataset_config is None:
            if "dataset_config" not in data:
                raise ValueError("dataset_config is required in config dict")
            dataset_config = dataset_config_from_dict(data["dataset_config"])

        if validation_dataset_config is None and "validation_dataset_config" in data:
            validation_dataset_config = dataset_config_from_dict(
                data["validation_dataset_config"]
            )

        return cls(
            model_name=data["model_name"],
            epochs=data["epochs"],
            learning_rate=data.get("learning_rate", 1e-3),
            batch_size=data.get("batch_size", 32),
            validate=data.get("validate", True),
            validation_batch_size=data.get("validation_batch_size"),
            optimizer_type=data.get("optimizer_type", "adam"),
            loss_fn_type=data.get("loss_fn_type", "cross_entropy"),
            dataset_config=dataset_config,
            validation_dataset_config=validation_dataset_config,
        )

    def save_json(self, filepath: Path):
        """Save training config to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls, filepath: Path, **kwargs) -> "TrainingConfig":
        """Load training config from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data, **kwargs)


def train_transformer(
    config: TrainingConfig,
    model_config: HookedTransformerConfig,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path = Path("out") / "models",
) -> torch.nn.Module:
    """Train a transformer using a TrainingConfig.

    Args:
        config: TrainingConfig with all training parameters including dataset configs.
        model_config: HookedTransformerConfig for the model.
        tokenizer: The tokenizer for the model.
        save_dir: Directory to save model in. The model is saved in a subdirectory
            named after config.model_name inside the specified save_dir.

    Returns:
        The trained model.

    Raises:
        ValueError: If config.validate is True but no validation_dataset_config provided.
        FileExistsError: If save_dir exists and is not empty.
    """

    # Initialize model from config
    from transformer_lens import HookedTransformer

    print("Initializing model from config...")
    model = HookedTransformer(model_config)

    # Create datasets from configs
    print("Creating training dataset...")
    training_dataset = config.dataset_config.create_dataset()
    training_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=config.batch_size
    )

    validation_loader = None
    if config.validate:
        if config.validation_dataset_config is None:
            raise ValueError(
                "validation_dataset_config must be provided when validate=True"
            )
        print("Creating validation dataset...")
        validation_dataset = config.validation_dataset_config.create_dataset()
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=config.validation_batch_size
        )

    # Setup optimizer
    if config.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

    # Setup loss function
    if config.loss_fn_type == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function type: {config.loss_fn_type}")

    # Prepare save directory
    save_dir = save_dir / config.model_name
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"Save directory {save_dir} is not empty.")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        for batch in training_loader:
            tokenized_batch = [tokenizer.encode(sample) for sample in batch]
            inputs = torch.tensor([item[:-1] for item in tokenized_batch])
            targets = torch.tensor([item[1:] for item in tokenized_batch])

            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        if config.validate and validation_loader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in validation_loader:
                    tokenized_batch = [tokenizer.encode(sample) for sample in batch]
                    inputs = torch.tensor([item[:-1] for item in tokenized_batch])
                    targets = torch.tensor([item[1:] for item in tokenized_batch])

                    logits = model(inputs)
                    logits = logits.view(-1, logits.size(-1))
                    targets = targets.view(-1)

                    val_loss = loss_fn(logits, targets)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(validation_loader)
            print(
                f"Epoch {epoch + 1}/{config.epochs}, Validation Loss: {avg_val_loss:.4f}"
            )

    # Save model and configs
    save_model(config, model, model_config, tokenizer, save_dir)

    return model


def save_model(
    config: TrainingConfig,
    model: torch.nn.Module,
    model_config: HookedTransformerConfig,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path,
) -> None:
    """Save model weights, model config, tokenizer, and training args to separate files.

    This keeps model_config.json and training_args.json separate as requested.

    Args:
        config: TrainingConfig with training parameters.
        model: The trained model.
        model_config: HookedTransformerConfig for the model.
        tokenizer: The tokenizer.
        save_dir: Directory to save all files.
    """

    # Save model weights
    weights_path = save_dir / "weights.pt"
    torch.save(model.state_dict(), weights_path)

    # Save model config (separate file)
    model_config_path = save_dir / "model_config.json"
    with model_config_path.open("w") as f:
        json.dump(
            model_config.to_dict(),
            f,
            indent=4,
            cls=subspace_partition.serialization.HookedTransformerConfigEncoder,
        )

    # Save tokenizer
    tokenizer_path = save_dir / "tokenizer.json"
    with tokenizer_path.open("w") as f:
        json.dump(
            tokenizer.to_dict(),
            f,
            indent=4,
        )

    # Save training args (separate file)
    training_args_path = save_dir / "training_args.json"
    config.save_json(training_args_path)

    print(f"Model saved to {save_dir}")
    print(f"  - weights.pt")
    print(f"  - model_config.json")
    print(f"  - tokenizer.json")
    print(f"  - training_args.json")
