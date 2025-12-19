import shutil
from typing import Literal
import faiss
from faiss import read_index, write_index
import os
import torch
import json
from tqdm import tqdm
from glob import glob
import numpy as np
import argparse
from pathlib import Path
from subspace_partition.preimage.utils import *


def run_build_index(
    experiment_name: str,
    distance_measure: Literal["euclidean", "cosine"] = "cosine",
    output_dir: Path = Path("out/index"),
    cached_acts_dir: Path = Path("out/preimage"),
    trained_Rs_dir: Path = Path("out/subspace_partition"),
    overwrite_existing: bool = False,
):
    """Build index for subspace partition preimage search.

    Args:
        trained_Rs_dir: Directory containing trained R matrices and configs.
        experiment_name: Name of the experiment (used for output directory).
        distance_measure: Distance measure to use.
        output_dir: Directory to save the built indices.
        cached_acts_dir: Directory containing cached activations (all cached acts, not just for this model
            the correct ones are loaded based on model name and site).
        trained_Rs_dir: Directory containing the outputs of the subspace partition training (all
            experiments, not just the one for this model, the correct ones are loaded based on
            the experiment_name argument).
        overwrite_existing: Whether to overwrite the output directory if it already exists."""
    
    torch.set_grad_enabled(False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    exp_dir = trained_Rs_dir / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Directory {exp_dir} not found.")

    output_dir = output_dir / f"index-{experiment_name}-{distance_measure}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if distance_measure == "euclidean":
        cosine = False
    else:
        cosine = True
    subtract_mean = False

    for file in glob("R*.pt", root_dir=exp_dir):
        _, model_name, site_name = file[:-3].split("-")

        output_dir_site = output_dir / f"{model_name}-{site_name}"
        if output_dir_site.exists():
            if overwrite_existing:
                shutil.rmtree(output_dir_site)
            else:
                raise ValueError(
                    "Output directory for this site already exists. Use override=True to overwrite or chose a different directory."
                )
        output_dir_site.mkdir(parents=True)

        R_path = exp_dir / file
        print("build index using R from", R_path)
        R = torch.load(R_path, map_location="cpu")["R.parametrizations.weight.0.base"]
        R = R.to(device)

        with open(exp_dir / f"R_config{file[1:-3]}.json") as f:
            partition = json.load(f)["partition"]

        if cosine:
            indices = [faiss.IndexFlatIP(p) for p in partition]
        else:
            indices = [faiss.IndexFlatL2(p) for p in partition]

        act_data_path = (
            cached_acts_dir / f"shared_acts-{model_name}" / f"{site_name}.pt"
        )
        acts = torch.load(act_data_path)

        batch_size = 1024
        norms = torch.empty(
            (acts.size(0), len(partition)), device=device, dtype=torch.float
        )

        if cosine and subtract_mean:
            random_idx = torch.randperm(acts.size(0))
            sum = 0
            total_num = 0
            for i in range(0, min(acts.size(0), 50_000), batch_size):
                rotated = (
                    acts[random_idx[i : i + batch_size]].to(device).to(R.dtype) @ R
                )
                sum += rotated.sum(dim=0)
                total_num += rotated.size(0)
            mean = (sum / total_num).unsqueeze(0)

        for i in tqdm(range(0, acts.size(0), batch_size)):
            rotated = acts[i : i + batch_size].to(device).to(R.dtype) @ R
            if cosine and subtract_mean:
                rotated -= mean
            for j, (chunk, index) in enumerate(
                zip(rotated.split(partition, dim=1), indices)
            ):
                chunk_norm = torch.linalg.vector_norm(chunk, dim=1, keepdim=True).clamp(
                    min=1e-8
                )
                norms[i : i + batch_size, j] = chunk_norm.squeeze(1)
                if cosine:
                    chunk /= chunk_norm
                index.add(chunk.cpu())

        # save indices
        print("saving indices...")
        for i, (index, p) in enumerate(zip(indices, partition)):
            assert index.d == p
            write_index(index, os.path.join(output_dir_site, f"{i}-{p}.index"))
        np.save(os.path.join(output_dir_site, "norms.npy"), norms.cpu().numpy())
