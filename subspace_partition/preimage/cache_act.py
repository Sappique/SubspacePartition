from transformer_lens import HookedTransformer
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
import torch
from tqdm import tqdm
import gc
import os
import pickle
import argparse
import re
from pathlib import Path
from collections import defaultdict
import gc
from subspace_partition.preimage.utils import *
import shutil
import random
import numpy as np
import json
from transformers import GPT2LMHeadModel
from subspace_partition.dataset_configs import DatasetConfig


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_cache_act(
    model_name: str,
    dataset_config: DatasetConfig,
    act_sites: list[str],
    output_dir: Path = Path("out/preimage"),
    max_in_memory: int = 10_000_000,
    batch_size: int = 32,
    override: bool = False,
):
    """Cache activations for a model on a dataset.

    Args:
        model_name: Name of the model to load from out/models/{model_name}.
        dataset_config: Configuration for the dataset to cache activations from.
        act_sites: List of activation sites to cache (e.g., ["blocks.0.hook_resid_post"]).
        output_dir: Directory to save cached activations.
        max_in_memory: Maximum number of activations to keep in memory before saving to disk.
        batch_size: Batch size to use for loading data.
        override: Whether to overwrite existing cache directory.
    """

    torch.set_grad_enabled(False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load model by name
    from subspace_partition.model_configs import load_model

    print(f"Loading model '{model_name}' from out/models/{model_name}")
    model = load_model(model_name, load_training_config=False)
    assert isinstance(model, HookedTransformer)

    # Create dataset from config
    print(f"Creating dataset from config...")
    dataset = dataset_config.create_dataset()

    save_dtype = torch.float16
    set_seed(0)

    save_dir = output_dir / f"shared_acts-{model_name}"
    if save_dir.exists():
        if override:
            shutil.rmtree(save_dir)
        else:
            raise ValueError(
                "Output directory already exists. Use override=True to overwrite or chose a different directory."
            )
    save_dir.mkdir(parents=True, exist_ok=False)

    json.dump(
        {
            "model_name": model_name,
            "dataset_config": dataset_config.to_dict(),
            "act_sites": act_sites,
        },
        open(save_dir / "cache_config.json", "w"),
        indent=4,
    )

    temp_dir = save_dir / "temp"
    temp_dir.mkdir()

    model.reset_hooks()

    cache = model.add_caching_hooks(act_sites)

    if all(site.startswith("blocks.") for site in act_sites):
        stop_at_layer = (
            max(
                int(re.findall(r"blocks\.(\d+)\.hook_resid_post", site)[0])
                for site in act_sites
            )
            + 1
        )
    else:
        stop_at_layer = None

    cached_act = defaultdict(list)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    cached_input = []
    total_count = 0
    split_count = 0
    for batch in tqdm(data_loader, desc="caching activations"):
        token_ids = model.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            max_length=model.cfg.n_ctx - 1,
            truncation=True,
        )["input_ids"]
        bos = torch.full(
            (token_ids.size(0), 1),
            fill_value=model.tokenizer.eos_token_id,
            dtype=token_ids.dtype,
        )
        token_ids = torch.cat([bos, token_ids], dim=1).to(device)

        with torch.autocast("cuda"):
            model(token_ids, return_type=None, stop_at_layer=stop_at_layer)

        m = token_ids != model.tokenizer.eos_token_id
        m[:, 0] = True  # keep bos
        for act_site in cache:
            cached_act[act_site].append(cache[act_site][m].to(save_dtype))

        def select(seq, mask):
            return [t for t, m_element in zip(seq, mask) if m_element]

        cached_input.extend(
            [
                model.tokenizer.convert_ids_to_tokens(select(seq, mask))
                for seq, mask in zip(token_ids.tolist(), m.tolist())
            ]
        )

        num_in_memory = len(cached_act) * sum(
            a.size(0) for a in next(iter(cached_act.values()))
        )
        if num_in_memory >= max_in_memory:
            print("saving batch to disk...")
            for act_site in cached_act:
                file_path = temp_dir / f"{act_site}-split{split_count}.pt"
                torch.save(torch.cat(cached_act[act_site], dim=0), file_path)
            split_count += 1
            cached_act = defaultdict(list)
            gc.collect()

        total_count += m.sum().item()

    # save last batch
    if any(len(cached_act[act_site]) > 0 for act_site in cached_act):
        for act_site in cached_act:
            file_path = temp_dir / f"{act_site}-split{split_count}.pt"
            torch.save(torch.cat(cached_act[act_site], dim=0), file_path)
        split_count += 1
        del cached_act
        gc.collect()

    print("loading and merging cached act...")
    print("total count", total_count)
    for act_site in tqdm(cache):
        merged_act = torch.zeros(
            (total_count, model.cfg.d_model), dtype=save_dtype, device=device
        )
        cursor = 0
        for i in range(split_count):
            act = torch.load(temp_dir / f"{act_site}-split{i}.pt")
            assert act.dtype == save_dtype
            merged_act[cursor : cursor + act.size(0)] = act
            cursor += act.size(0)
            os.remove(temp_dir / f"{act_site}-split{i}.pt")
        torch.save(
            merged_act, save_dir / f"{site_name_to_short_name(act_site)}.pt"
        )  # always use short name in file system
    temp_dir.rmdir()

    print("saving input...")
    with open(os.path.join(save_dir, "str_tokens.pkl"), "wb") as f:
        pickle.dump(cached_input, f, protocol=pickle.HIGHEST_PROTOCOL)
