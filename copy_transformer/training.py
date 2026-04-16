import json
import warnings
from pathlib import Path
from typing import Literal, Any
from dataclasses import dataclass, field
import torch
import infra.serialization
from infra import OUT_DIR

from transformers import PreTrainedTokenizerBase
from transformer_lens import HookedTransformerConfig
from infra.dataset_configs import DatasetConfig


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
        from infra.dataset_configs import dataset_config_from_dict

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
    save_dir: Path = OUT_DIR / "models",
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
    model.set_tokenizer(tokenizer)

    # Custom collate function that handles both plain strings and (string, mask) tuples
    # without trying to stack variable-length masks into tensors
    def collate_fn(batch):
        return list(batch)

    # Create datasets from configs
    print("Creating training dataset...")
    training_dataset = config.dataset_config.create_dataset()
    training_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=config.batch_size, collate_fn=collate_fn
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
            validation_dataset,
            batch_size=config.validation_batch_size,
            collate_fn=collate_fn,
        )

    # Setup optimizer
    if config.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

    # Setup loss function (with reduction='none' to allow masking padding tokens)
    if config.loss_fn_type == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    else:
        raise ValueError(f"Unsupported loss function type: {config.loss_fn_type}")

    # Prepare save directory
    save_dir = save_dir / config.model_name
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"Save directory {save_dir} is not empty.")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Get context length and padding token (use model's tokenizer to match model.to_tokens())
    context_length = model_config.n_ctx
    if hasattr(model.tokenizer, "pad_token_id") and isinstance(
        model.tokenizer.pad_token_id, int
    ):
        pad_token_id: int = model.tokenizer.pad_token_id
    else:
        raise ValueError("Tokenizer must have a pad_token_id attribute.")

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        for batch in training_loader:
            # Check if batch contains dicts (with mask) or plain strings
            if isinstance(batch[0], dict):
                sequences = [item["text"] for item in batch]
                custom_masks = [item["mask"] for item in batch]
                has_custom_mask = True
            else:
                sequences = list(batch)
                custom_masks = None
                has_custom_mask = False

            if any(len(prompt) > context_length - 1 for prompt in sequences):
                warnings.warn(
                    f"Some sequences in the training set are longer than"
                    f" {context_length - 1} (context length - 1 for the BOS"
                    f" token), they will be truncated."
                )

            # Tokenize with BOS token using model.to_tokens()
            # model.to_tokens() returns tensor of shape (batch, seq_len) with BOS prepended
            tokens = model.to_tokens(sequences, prepend_bos=True)

            # Check for sequences that exceed context length and truncate
            if tokens.shape[1] > context_length:
                num_truncated = tokens.shape[1] > context_length
                warnings.warn(
                    f"Truncating {tokens.shape[0]} sequence(s) from length {tokens.shape[1]} "
                    f"to context length {context_length}"
                )
                tokens = tokens[:, :context_length]

            # Pad sequences shorter than context length
            if tokens.shape[1] < context_length:
                padding_length = context_length - tokens.shape[1]
                padding = torch.full(
                    (tokens.shape[0], padding_length),
                    pad_token_id,
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                tokens = torch.cat([tokens, padding], dim=1)

            # Create inputs (all but last token) and targets (all but first token)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            # Create mask to ignore padding tokens in loss (1 for real tokens, 0 for padding)
            # We mask positions where the TARGET is a padding token
            target_mask = (targets != pad_token_id).float()

            # Apply custom mask if provided (for masking first pattern repetition)
            if has_custom_mask:
                assert custom_masks is not None  # Type narrowing for type checker
                # Convert character-level masks to tensors
                # Each mask needs to be padded/truncated to match target length
                # Account for: BOS token prepended, then shifted for targets
                # Target positions correspond to predicting tokens at positions 1, 2, ..., context_length-1
                # Custom mask is for positions 0, 1, ..., len(sequence)-1
                # After BOS prepend and shift: target[i] predicts token at position i+1 in original sequence
                # So custom_mask[i] should apply to target[i] (predicting position i in sequence, 0-indexed)
                batch_size = tokens.shape[0]
                target_len = targets.shape[1]
                custom_mask_tensor = torch.zeros(
                    batch_size, target_len, device=tokens.device
                )

                for i, mask in enumerate(custom_masks):
                    # mask corresponds to sequence positions 0..len(seq)-1
                    # targets correspond to predicting positions 1..context_length-1 after BOS
                    # So target[j] predicts position j in the original sequence (0-indexed)
                    mask_len = min(len(mask), target_len)
                    custom_mask_tensor[i, :mask_len] = torch.tensor(
                        mask[:mask_len], dtype=torch.float32, device=tokens.device
                    )

                # Combine with padding mask (both must be 1 for position to be included)
                target_mask = target_mask * custom_mask_tensor

            logits = model(inputs)

            # Compute loss with masking for padding tokens
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            mask_flat = target_mask.reshape(-1)

            # Compute per-token loss and apply mask
            per_token_loss = loss_fn(logits_flat, targets_flat)
            masked_loss = per_token_loss * mask_flat
            loss = masked_loss.sum() / mask_flat.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        if config.validate and validation_loader is not None:
            model.eval()
            total_val_loss = 0.0
            total_val_tokens = 0
            with torch.no_grad():
                for batch in validation_loader:
                    # Check if batch contains dicts (with mask) or plain strings
                    if isinstance(batch[0], dict):
                        sequences = [item["text"] for item in batch]
                        custom_masks = [item["mask"] for item in batch]
                        has_custom_mask = True
                    else:
                        sequences = list(batch)
                        custom_masks = None
                        has_custom_mask = False

                    # Tokenize with BOS token using model.to_tokens()
                    tokens = model.to_tokens(sequences, prepend_bos=True)

                    # Truncate if needed
                    if tokens.shape[1] > context_length:
                        tokens = tokens[:, :context_length]

                    # Pad if needed
                    if tokens.shape[1] < context_length:
                        padding_length = context_length - tokens.shape[1]
                        padding = torch.full(
                            (tokens.shape[0], padding_length),
                            pad_token_id,
                            dtype=tokens.dtype,
                            device=tokens.device,
                        )
                        tokens = torch.cat([tokens, padding], dim=1)

                    inputs = tokens[:, :-1]
                    targets = tokens[:, 1:]

                    # Create mask to ignore padding tokens in loss
                    target_mask = (targets != pad_token_id).float()

                    # Apply custom mask if provided
                    if has_custom_mask:
                        assert (
                            custom_masks is not None
                        )  # Type narrowing for type checker
                        batch_size = tokens.shape[0]
                        target_len = targets.shape[1]
                        custom_mask_tensor = torch.zeros(
                            batch_size, target_len, device=tokens.device
                        )

                        for i, mask in enumerate(custom_masks):
                            mask_len = min(len(mask), target_len)
                            custom_mask_tensor[i, :mask_len] = torch.tensor(
                                mask[:mask_len],
                                dtype=torch.float32,
                                device=tokens.device,
                            )

                        target_mask = target_mask * custom_mask_tensor

                    logits = model(inputs)

                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = targets.reshape(-1)
                    mask_flat = target_mask.reshape(-1)

                    per_token_loss = loss_fn(logits_flat, targets_flat)
                    masked_loss = per_token_loss * mask_flat
                    total_val_loss += masked_loss.sum().item()
                    total_val_tokens += mask_flat.sum().item()

            avg_val_loss = total_val_loss / max(total_val_tokens, 1)
            print(
                f"Epoch {epoch + 1}/{config.epochs}, Validation Loss: {avg_val_loss:.4f}"
            )

    # Save model and configs
    save_model(config, model, model_config, save_dir)

    return model


def save_model(
    config: TrainingConfig,
    model: torch.nn.Module,
    model_config: HookedTransformerConfig,
    save_dir: Path,
) -> None:
    """Save model weights, model config, tokenizer, and training args to separate files.

    Args:
        config: TrainingConfig with training parameters.
        model: The trained model.
        model_config: HookedTransformerConfig for the model.
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
            cls=infra.serialization.HookedTransformerConfigEncoder,
        )

    # Save tokenizer
    tokenizer_path = save_dir / "tokenizer.json"
    with tokenizer_path.open("w") as f:
        json.dump(
            model.tokenizer.to_dict(),
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
