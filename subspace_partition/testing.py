import torch
from typing import Literal

import subspace_partition.model_configs
from subspace_partition.dataset_configs import DatasetConfig


def test_model_performance(
    model_name: str,
    dataset_config: DatasetConfig | None = None,
    test_n_last_tokens: int = 1,
    metric: Literal["cross_entropy", "accuracy"] = "accuracy",
    num_samples: int = 1000,
    batch_size: int = 32,
) -> dict[str, float]:
    """Test a model's performance on a dataset.

    Args:
        model_name: Name of the model to test.
        dataset_config: Dataset configuration to use for testing.
            If None, uses the model's training dataset.
        test_n_last_tokens: Number of last tokens in each prompt to evaluate.
            For example, if test_n_last_tokens=3 and prompt is "ABCABC",
            the model is evaluated on predicting the last 3 tokens ("ABC").
        metric: Metric to compute. Either "cross_entropy" or "accuracy".
        num_samples: Number of samples to test on.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary with metric results:
            - "cross_entropy": Average cross-entropy loss on last n tokens
            - "accuracy": Percentage of correctly predicted tokens (0-100)
            - "num_samples": Number of samples evaluated
            - "num_tokens": Total number of tokens evaluated
    """
    model, training_config = subspace_partition.model_configs.load_model(
        model_name, load_training_config=True
    )

    if dataset_config is None:
        dataset_config = training_config.dataset_config

    dataset = dataset_config.create_dataset()

    # Custom collate function to handle both plain strings and dicts
    def collate_fn(batch):
        return list(batch)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn
    )

    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    samples_seen = 0

    with torch.no_grad():
        for batch in data_loader:
            if samples_seen >= num_samples:
                break

            # Extract text from dicts if needed
            if isinstance(batch[0], dict):
                sequences = [item["text"] for item in batch]
            else:
                sequences = list(batch)

            # Filter out sequences that are too short
            valid_sequences = [
                seq for seq in sequences if len(seq) >= test_n_last_tokens + 1
            ]
            if not valid_sequences:
                continue

            # Tokenize with BOS token
            tokens = model.to_tokens(valid_sequences, prepend_bos=True)
            tokens = tokens.to(device)

            # Create inputs (all but last token) and targets (all but first token)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            # Get logits
            logits = model(inputs)

            # Only evaluate on the last n tokens of each sequence
            # Sequence length after tokenization (excluding BOS in target)
            seq_lengths = (tokens != model.tokenizer.pad_token_id).sum(dim=1) - 1

            for i in range(len(valid_sequences)):
                seq_len = seq_lengths[i].item()
                n_tokens = min(test_n_last_tokens, seq_len)

                if n_tokens <= 0:
                    continue

                # Get the last n token positions
                start_idx = int(seq_len - n_tokens)
                end_idx = int(seq_len)

                seq_logits = logits[i, start_idx:end_idx]  # (n_tokens, vocab_size)
                seq_targets = targets[i, start_idx:end_idx]  # (n_tokens,)

                if metric == "cross_entropy" or metric == "accuracy":
                    # Compute cross-entropy loss
                    loss = torch.nn.functional.cross_entropy(
                        seq_logits, seq_targets, reduction="sum"
                    )
                    total_loss += loss.item()

                if metric == "accuracy" or metric == "cross_entropy":
                    # Compute accuracy
                    predictions = seq_logits.argmax(dim=-1)
                    correct = (predictions == seq_targets).sum().item()
                    total_correct += correct

                total_tokens += n_tokens

            samples_seen += len(valid_sequences)

    results = {
        "num_samples": samples_seen,
        "num_tokens": total_tokens,
    }

    if total_tokens > 0:
        if metric == "cross_entropy":
            results["cross_entropy"] = total_loss / total_tokens
        if metric == "accuracy":
            results["accuracy"] = 100.0 * total_correct / total_tokens
    else:
        results["cross_entropy"] = float("nan")
        results["accuracy"] = float("nan")

    return results


def print_test_results(results: dict[str, float]) -> None:
    """Pretty-print test results.

    Args:
        results: Dictionary returned by test_model_performance.
    """
    print(f"Samples evaluated: {results['num_samples']}")
    print(f"Tokens evaluated: {results['num_tokens']}")

    if "accuracy" in results:
        print(f"Accuracy: {results['accuracy']:.2f}%")
    if "cross_entropy" in results:
        print(f"Cross-entropy: {results['cross_entropy']:.4f}")
