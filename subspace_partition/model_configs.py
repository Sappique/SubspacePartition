import json
from pathlib import Path

import torch
import transformer_lens
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from transformer_lens.pretrained.weight_conversions.gpt2 import convert_gpt2_weights
import copy_transformer.tokenizer
import copy_transformer.training
import subspace_partition.serialization
from subspace_partition.dataset_configs import UniqueNgramPatternConfig

CUSTOM_MODELS_DIR = Path(__file__).parent.parent / "out" / "models"

# HuggingFace GPT2 models from lacoco-lab/decompiling_transformers.
# Maps model name prefix to a dict defining the tokenizer vocab, HookedTransformerConfig kwargs,
# and a reconstructed dataset config matching the original training distribution.
_HF_MODEL_CONFIGS = {
    "unique_bigram_copy": {
        # 16 data tokens ("0"-"15"), plus 4 special tokens
        "data_tokens": [str(i) for i in range(16)],
        "special_tokens": {
            "bos": "<bos>",
            "sep": "<sep>",
            "eos": "<eos>",
            "pad": "<pad>",
        },
        "hooked_transformer_kwargs": dict(
            n_layers=2,
            n_heads=4,
            d_model=256,
            d_head=64,
            n_ctx=303,
            d_vocab=20,
            act_fn="gelu_new",
            original_architecture="GPT2LMHeadModel",
            tokenizer_name=None,
        ),
        # Reconstructed from lacoco-lab/decompiling_transformers source:
        # - trained on lengths 0-50, evaluated up to 150
        # - 16 data tokens, unique bigrams (n=2), <sep> between repetitions
        # - token_separator=" " because the tokenizer uses WhitespaceSplit
        "dataset_config": UniqueNgramPatternConfig(
            vocabulary=[str(i) for i in range(16)],
            n=2,
            separator="<sep>",
            max_pattern_length=50,
            min_pattern_length=2,
            mask_first_repetition=True,
            token_separator=" ",
            iterable=True,
            length="infinite",
        ),
        "training_kwargs": dict(
            epochs=3,
            learning_rate=1e-3,
            batch_size=64,
        ),
    },
}


def _get_hf_config(model_name: str) -> dict | None:
    """Return the HF model config if model_name matches a known prefix, else None."""
    for prefix, config in _HF_MODEL_CONFIGS.items():
        if model_name.startswith(prefix):
            return config
    return None


def _build_hf_tokenizer(config: dict) -> PreTrainedTokenizerFast:
    """Build a PreTrainedTokenizerFast from an _HF_MODEL_CONFIGS entry."""
    special = config["special_tokens"]
    vocab = {tok: i for i, tok in enumerate(config["data_tokens"])}
    n_data = len(vocab)
    vocab[special["bos"]] = n_data
    vocab[special["sep"]] = n_data + 1
    vocab[special["eos"]] = n_data + 2
    vocab[special["pad"]] = n_data + 3

    tok_model = models.WordLevel(vocab=vocab, unk_token=special["pad"])
    base_tokenizer = Tokenizer(tok_model)
    base_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer,
        bos_token=special["bos"],
        eos_token=special["eos"],
        pad_token=special["pad"],
        sep_token=special["sep"],
    )
    tokenizer.init_kwargs["name_or_path"] = "custom"
    tokenizer.init_kwargs["add_bos_token"] = True
    tokenizer.add_bos_token = True
    return tokenizer


def _load_hf_model(
    model_path: Path,
    hf_config: dict,
    device: torch.device,
) -> transformer_lens.HookedTransformer:
    """Load a HuggingFace GPT2LMHeadModel and convert to HookedTransformer."""
    hf_model = GPT2LMHeadModel.from_pretrained(model_path)

    cfg = transformer_lens.HookedTransformerConfig(
        **hf_config["hooked_transformer_kwargs"]
    )

    state_dict = convert_gpt2_weights(hf_model, cfg)
    tokenizer = _build_hf_tokenizer(hf_config)

    model = transformer_lens.HookedTransformer(
        cfg, tokenizer=tokenizer, move_to_device=False
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_model(
    model_name: str,
    load_training_config: bool = False,
    device: torch.device | str | None = None,
) -> (
    transformer_lens.HookedTransformer
    | tuple[
        transformer_lens.HookedTransformer, copy_transformer.training.TrainingConfig
    ]
):
    """Load a model by name.

    Supports two model formats:
    - Custom models (model_config.json + tokenizer.json + weights.pt)
    - HuggingFace GPT2 models from lacoco-lab/decompiling_transformers
      (config.json + model.safetensors), identified by name prefix.

    Args:
        model_name: The name of the model to load (directory name under out/models/).
        load_training_config: If True, also loads and returns the TrainingConfig.
            For HF pretrained models, a TrainingConfig is reconstructed from
            known training parameters.
        device: Device to load the model onto. If None, uses CUDA if available, else CPU.

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

    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Check if this is a known HuggingFace model
    hf_config = _get_hf_config(model_name)
    if hf_config is not None:
        model = _load_hf_model(model_path, hf_config, device)
        if load_training_config:
            training_config = copy_transformer.training.TrainingConfig(
                model_name=model_name,
                dataset_config=hf_config["dataset_config"],
                **hf_config["training_kwargs"],
            )
            return model, training_config
        return model

    # Otherwise, load as custom model
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
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()

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
