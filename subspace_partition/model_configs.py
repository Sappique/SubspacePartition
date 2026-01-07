import json
from pathlib import Path

import torch
import transformer_lens
import copy_transformer.tokenizer
import copy_transformer.training
import subspace_partition.serialization

CUSTOM_MODELS_DIR = Path(__file__).parent.parent / "out" / "models"


def load_model(
    model_name: str,
    load_training_config: bool = False,
) -> (
    transformer_lens.HookedTransformer
    | tuple[
        transformer_lens.HookedTransformer, copy_transformer.training.TrainingConfig
    ]
):
    """Load a custom model by name.

    Args:
        model_name: The name of the custom model to load.
        load_training_config: If True, also loads and returns the TrainingConfig.

    Returns:
        If load_training_config is False: The loaded HookedTransformer model.
        If load_training_config is True: Tuple of (model, training_config).

    Raises:
        ValueError: If the model name is not recognized.
        FileNotFoundError: If required files are missing.
    """

    model_path = CUSTOM_MODELS_DIR / model_name
    if not model_path.exists():
        raise ValueError(f"Model '{model_name}' not found in custom models directory.")

    model_config_path = model_path / "model_config.json"
    tokenizer_config_path = model_path / "tokenizer.json"
    model_weights_path = model_path / "weights.pt"
    training_args_path = model_path / "training_args.json"

    # Load model config
    with open(model_config_path, "r") as f:
        model_config = transformer_lens.HookedTransformerConfig(
            **json.load(
                f,
                object_hook=subspace_partition.serialization.hooked_transformer_config_decoder,
            )
        )

    # Create and load model
    model = transformer_lens.HookedTransformer(model_config)
    model.load_state_dict(torch.load(model_weights_path))

    # Load tokenizer
    with open(tokenizer_config_path, "r") as f:
        model.set_tokenizer(
            copy_transformer.tokenizer.SingleCharTokenizer.from_dict(json.load(f))
        )

    # Optionally load training config
    if load_training_config:
        if not training_args_path.exists():
            raise FileNotFoundError(
                f"training_args.json not found for model '{model_name}'. "
                "This model may have been trained with an older version."
            )
        training_config = copy_transformer.training.TrainingConfig.from_json(
            training_args_path
        )
        return model, training_config

    return model
