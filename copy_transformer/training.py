from typing import Literal
import torch

from transformers import PreTrainedTokenizerBase


def train_transformer(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    training_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader | None,
    epochs: int,
    validate: bool = True,
    optimizer: torch.optim.Optimizer | Literal["adam"] = "adam",
    learning_rate: float = 1e-3,
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
) -> None:
    """Train the given transformer using the provided data loaders and parameters.

    Args:
        model: The transformer model to be trained.
        tokenizer: The tokenizer for the model.
        training_loader: DataLoader for the training dataset.
        validation_loader: DataLoader for the validation dataset. Can be None if validate is False.
        epochs: Number of epochs to train the model.
        validate: Whether to perform validation after each epoch. Defaults to True.
        optimizer: Optimizer to use for training. Defaults to Adam optimizer.
        learning_rate: Learning rate for the optimizer. Defaults to 1e-3.
        loss_fn: Loss function to use. Defaults to CrossEntropyLoss.
    """

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if validate and validation_loader is None:
        raise ValueError(
            "Validation loader must be provided if validate is set to True."
        )

    for epoch in range(epochs):
        model.train()
        for batch in training_loader:

            tokenized_batch = [tokenizer(sample).input_ids for sample in batch]
            inputs = torch.tensor([item[:-1] for item in tokenized_batch])
            targets = torch.tensor([item[1:] for item in tokenized_batch])

            logits = model(inputs)

            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if validate and validation_loader is not None:
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
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")


def add_hooks(model, layers):
    """
    model: the PyTorch model
    layers: list of module names (strings) to save activations from

    returns: (activation_dict, hook_handles)
    """

    activations = {}

    def hook_fn(name):
        def fn(module, input, output):
            # Always detach and move to CPU so we don't keep the graph
            output, _ = output
            activations[name] = output.detach().cpu()

        return fn

    handles = []
    for name, module in model.named_modules():
        if name in layers:
            h = module.register_forward_hook(hook_fn(name))
            handles.append(h)

    return activations, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


@torch.no_grad
def log_activations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    loader: torch.utils.data.DataLoader,
) -> dict[str, list[torch.Tensor]]:
    """
    Logs the activations of attention 1 and 2 for the given model and data loader.
    """
    model.eval()
    activations = {"attention1": [], "attention2": []}
    current_activations, hooks = add_hooks(model, ["attention1", "attention2"])
    for batch in loader:
        tokenized_batch = [tokenizer.encode(sample) for sample in batch]
        inputs = torch.tensor([item[:-1] for item in tokenized_batch])
        model(inputs)
        activations["attention1"].append(current_activations["attention1"])
        activations["attention2"].append(current_activations["attention2"])

    remove_hooks(hooks)
    return activations
