from random import random
import subspace_partition.model_configs
import circuitsvis as cv


def show_model_attention_patterns(model_name: str, prompt: str | None = None) -> None:
    """Visualize attention patterns of a model.

    Args:
        model_name: Name of the model to visualize.
        prompt: Optional prompt to use for visualization.
            If None, the prompt will be taken from the dataset used during training.

    Has to be used in a Jupyter notebook environment for display to work.
    If no prompt is provided, the function will attempt to load a prompt
    from the dataset used during training. If the dataset implements __len__,
    a random prompt will be selected; otherwise, the first prompt will be used.
    """

    model, training_config = subspace_partition.model_configs.load_model(
        model_name, load_training_config=True
    )

    dataset_config = training_config.dataset_config
    dataset = dataset_config.create_dataset()

    if prompt is None:
        try:
            dataset_len = len(dataset)
            prompt = dataset[random.randint(0, dataset_len - 1)]
        except:
            prompt = dataset[0]

    prompt_str = prompt if prompt is not None else dataset[0]

    prompt_token = model.to_tokens(prompt_str)

    _, cache = model.run_with_cache(prompt_token, remove_batch_dim=True)

    prompt_separate_str_token = model.to_str_tokens(prompt_str)

    for layer in range(model.cfg.n_layers):
        attention_patterns = cache["pattern", layer, "attn"]
        display(
            cv.attention.attention_patterns(
                prompt_separate_str_token, attention_patterns
            )
        )
