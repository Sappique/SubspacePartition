from dataclasses import dataclass
from typing import Any, Literal
from subspace_partition.training.data import *
from subspace_partition.training.model import *
from subspace_partition.training.utils import *
from functools import partial
from matplotlib import pyplot as plt
from collections import defaultdict
from transformer_lens import HookedTransformerConfig
from transformers import PreTrainedTokenizerBase
from subspace_partition.dataset_configs import DatasetConfig, dataset_config_from_dict
import json

SUBSPACE_PARTITION_DIR = Path(__file__).parent.parent / "out" / "subspace_partition"


@dataclass
class SubspacePartitionConfig:
    """Configuration class for configuring the subspace partitioning method.

    Args:
        exp_name: Name of the experiment.
        model_name: Name of the model (e.g., "gpt2").
        dataset_config: Configuration for the dataset to use for training.
        act_sites: List of activation sites to train on (e.g., ["blocks.0.hook_resid_post"]).
        metric: Distance metric to use for nearest neighbor search.
        unit_size: Initial size of each subspace unit. The hidden dimension is divided
            into (h_dim // unit_size) subspaces of this size.
        max_steps: Total number of training iterations for the rotation matrix R.
        merge_interval: Number of steps between merge attempts, starting from merge_start.
        merge_start: Training step at which to begin attempting subspace merges.
        merge_thr: Threshold for merging subspaces. Subspace pairs with normalized MI
            above this threshold are candidates for merging.
        merge_metric: Metric to use for deciding which subspaces to merge. Currently only
            "mi" (mutual information) is implemented.
        acc_steps: Gradient accumulation steps. Gradients are accumulated over this many
            forward/backward passes before updating weights. Effective batch size = batch_size * acc_steps.
        batch_size: Number of query activations used in each training step. Those are the
            activations for which we find nearest neighbors in the buffer.
        test_batch_size: Batch size used during evaluation and merge metric computation
            (not training).
        search_steps: Number of key batches to search through when finding nearest neighbors.
            Total search pool per training step = search_steps * block_len. Higher values
            improve nearest neighbor approximation but slow training.
        block_len: Number of key activations used in each search step. Total search pool per
            training step = search_steps * block_len.
        lr: Learning rate for Adam optimizer.
        adam_beta1: Beta1 parameter for Adam optimizer.
        adam_beta2: Beta2 parameter for Adam optimizer.
        weight_type: Type of weight (e.g., "none").
        clip_grad: Maximum gradient norm for gradient clipping.
        device: Device to run training on (defaults to CUDA if available).
        output_dir: Directory to save trained models and logs.
        stop_at_n_subspaces: If set to an int, training will stop at the beginning of a
            merge step if the number of remaining subspaces is less than or equal to this value.

    The arguments do not straightforwardly explain how many steps are taken / data is used.
    Their relationship to that is as follows:

    There are `max_steps` many *training steps*.
    In each training step, `batch_size` many *query activations* are sampled from the dataset.
    For each query activation, the nearest neighbor is found among `search_steps` many *key batches*,
    each of size `block_len`. Thus, the total number of *key activations* to search through per
    training step is `search_steps * block_len`.
    """

    exp_name: str
    model_name: str
    dataset_config: DatasetConfig
    act_sites: list[str]
    metric: Literal["euclidean", "cosine"] = "euclidean"
    unit_size: int = 32
    max_steps: int = 50_000
    merge_interval: int = 3_000
    merge_start: int = 10_000
    merge_thr: float = 0.04
    merge_metric: Literal["mi"] = "mi"
    stop_at_n_subspaces: int | None = None
    acc_steps: int = 1
    batch_size: int = 128
    test_batch_size: int = 128
    search_steps: int = 25
    block_len: int = 16384
    lr: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_type: str = "none"
    clip_grad: float = 100.0
    device: torch.device | None = None
    output_dir: Path = SUBSPACE_PARTITION_DIR

    def __post_init__(self):
        self.refresh_block_num: int = 2048 * 2048 // self.block_len
        self.caching_batch_size: int = 16

        self.act_site: str | None = None

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset: Any | None = None  # Created from dataset_config at runtime

    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        config_dict = {
            "exp_name": self.exp_name,
            "model_name": self.model_name,
            "act_sites": self.act_sites,
            "batch_size": self.batch_size,
            "test_batch_size": self.test_batch_size,
            "acc_steps": self.acc_steps,
            "metric": self.metric,
            "max_steps": self.max_steps,
            "merge_interval": self.merge_interval,
            "merge_start": self.merge_start,
            "merge_thr": self.merge_thr,
            "merge_metric": self.merge_metric,
            "stop_at_n_subspaces": self.stop_at_n_subspaces,
            "search_steps": self.search_steps,
            "unit_size": self.unit_size,
            "lr": self.lr,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "weight_type": self.weight_type,
            "block_len": self.block_len,
            "clip_grad": self.clip_grad,
        }

        # Handle device
        config_dict["device"] = str(self.device)

        # Handle dataset config
        config_dict["dataset_config"] = self.dataset_config.to_dict()

        # Handle output_dir
        config_dict["output_dir"] = str(self.output_dir)

        return config_dict

    def save_json(self, filepath: str | Path):
        """Save configuration to a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(
        cls, config_dict: dict, dataset_config: DatasetConfig | None = None
    ) -> "SubspacePartitionConfig":
        """Create a SubspacePartitionConfig from a dictionary."""
        # Reconstruct device
        device = None
        if "device" in config_dict:
            device = torch.device(config_dict["device"])

        # Reconstruct dataset config if not provided
        if dataset_config is None:
            if "dataset_config" not in config_dict:
                raise ValueError("dataset_config is required in config dict")
            dataset_config = dataset_config_from_dict(config_dict["dataset_config"])

        # Create config instance
        return cls(
            exp_name=config_dict["exp_name"],
            model_name=config_dict["model_name"],
            dataset_config=dataset_config,
            act_sites=config_dict["act_sites"],
            batch_size=config_dict.get("batch_size", 128),
            test_batch_size=config_dict.get("test_batch_size", 128),
            acc_steps=config_dict.get("acc_steps", 1),
            metric=config_dict.get("metric", "euclidean"),
            max_steps=config_dict.get("max_steps", 50_000),
            merge_interval=config_dict.get("merge_interval", 3_000),
            merge_start=config_dict.get("merge_start", 10_000),
            merge_thr=config_dict.get("merge_thr", 0.04),
            merge_metric=config_dict.get("merge_metric", "mi"),
            stop_at_n_subspaces=config_dict.get("stop_at_n_subspaces", None),
            search_steps=config_dict.get("search_steps", 25),
            unit_size=config_dict.get("unit_size", 32),
            lr=config_dict.get("lr", 3e-4),
            adam_beta1=config_dict.get("adam_beta1", 0.9),
            adam_beta2=config_dict.get("adam_beta2", 0.999),
            weight_type=config_dict.get("weight_type", "none"),
            block_len=config_dict.get("block_len", 16384),
            clip_grad=config_dict.get("clip_grad", 100.0),
            device=device,
            output_dir=config_dict.get("output_dir", SUBSPACE_PARTITION_DIR),
        )

    @classmethod
    def from_json(
        cls, filepath: str | Path, dataset_config: DatasetConfig | None = None
    ) -> "SubspacePartitionConfig":
        """Load configuration from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict, dataset_config=dataset_config)


def run_subspace_partition(cfg: SubspacePartitionConfig):
    """Run subspace partition analysis on a model loaded by name."""
    set_seed(0)

    output_dir: Path = cfg.output_dir / cfg.exp_name

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if any(output_dir.iterdir()):
        raise ValueError(f"Output directory {output_dir} is not empty.")

    config_path = output_dir / "training_args.json"
    cfg.save_json(config_path)

    test_search_steps = max(1, 200 * 2048 // cfg.block_len)
    if cfg.unit_size <= 4:
        mi_search_steps = max(1, 5 * 2048 // cfg.block_len)
    else:
        mi_search_steps = max(1, 50 * 2048 // cfg.block_len)

    device = cfg.device

    # Load model by name (don't load training config)
    from subspace_partition.model_configs import load_model
    from transformer_lens import HookedTransformer

    print(f"Loading model '{cfg.model_name}' from out/models/{cfg.model_name}")
    model_result = load_model(cfg.model_name, load_training_config=False, device=device)
    # Type assertion since we know load_training_config=False returns just the model
    assert isinstance(model_result, HookedTransformer)
    hooked_model: HookedTransformer = model_result

    model_name = cfg.model_name
    h_dim = hooked_model.cfg.d_model

    # Create dataset from config
    print(f"Creating dataset from config...")
    dataset = cfg.dataset_config.create_dataset()
    cfg.dataset = dataset  # Store for BufferReuse

    for act_site in cfg.act_sites:
        print("training for", act_site)
        cfg.act_site = act_site
        site_name = site_name_to_short_name(act_site)

        if (output_dir / f"R-{model_name}-{site_name}.pt").exists():
            continue
        log_path = output_dir / f"train_log-{model_name}-{site_name}.txt"
        f = open(log_path, "w")
        print_ = partial(print_to_both, f=f)

        buffer = BufferReuse(cfg, hooked_model)

        R = NewUnevenRTrainer(
            h_dim, [cfg.unit_size] * (h_dim // cfg.unit_size), cfg, buffer
        ).to(cfg.device)

        optimizer = torch.optim.Adam(
            R.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2)
        )
        log_metrics = defaultdict(list)

        for i in tqdm(range(cfg.max_steps)):

            loss = R.step()

            loss.backward()

            if (i + 1) % cfg.acc_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    R.parameters(), max_norm=float("inf")
                )
                log_metrics["R_grad_norm"].append(grad_norm.item())
                torch.nn.utils.clip_grad_norm_(R.parameters(), max_norm=cfg.clip_grad)
                optimizer.step()
                optimizer.zero_grad()

            log_metrics["training_loss"].append(loss.item())

            if (i + 1) % 200 == 0:
                print_({k: sum(v) / len(v) for k, v in log_metrics.items()})
                log_metrics = defaultdict(list)

            if (
                (i + 1) >= cfg.merge_start
                and ((i + 1 - cfg.merge_start) % cfg.merge_interval == 0)
                and (i + 1) < (cfg.max_steps - 100)
            ):
                # Check if we should stop due to reaching target number of subspaces
                if (
                    cfg.stop_at_n_subspaces is not None
                    and len(R.partition) <= cfg.stop_at_n_subspaces
                ):
                    print_(
                        f"******* STOPPING: reached target of {cfg.stop_at_n_subspaces} subspace(s) "
                        f"(currently {len(R.partition)} subspaces with sizes {R.partition})"
                    )
                    break

                eval_result = []
                for j in tqdm(range(max(1, 50 * 128 // cfg.test_batch_size))):
                    eval_result.append(
                        R.evaluate_step(
                            num_steps=test_search_steps, batch_size=cfg.test_batch_size
                        )
                    )
                eval_result = torch.stack(eval_result).mean(dim=0)
                print_("eval result", eval_result)

                pairs = list(combinations(range(len(R.partition)), 2))

                print_("computing merge metric")
                mi = 0
                subspace_var = R.compute_subspace_var(num=2000)

                step = max(1, 100 * 128 // cfg.test_batch_size)
                for j in tqdm(range(step)):
                    if cfg.merge_metric == "mi":
                        mi_batch = R.compute_MI_step(
                            metric="euclidean",
                            pairs=pairs,
                            num_steps=mi_search_steps,
                            batch_size=cfg.test_batch_size,
                            subspace_var=subspace_var,
                        )
                    mi += mi_batch

                mi /= step
                metric = {}
                for pair_idx, (j, k) in enumerate(pairs):
                    metric[(j, k)] = mi[pair_idx].item() / (
                        R.partition[j] + R.partition[k]
                    )

                lis = sorted([(k, v) for k, v in metric.items()], key=lambda x: -x[1])
                if len(lis) > 300:
                    print_("sorted normed mi top 10", lis[:10])
                    print_("sorted normed mi last 10", lis[-10:])
                else:
                    print_("normed mi", lis)

                covered = set()
                pairs_to_merge = []
                for k, v in lis:
                    if (
                        v > cfg.merge_thr
                        and k[0] not in covered
                        and k[1] not in covered
                    ):
                        pairs_to_merge.append(k)
                        covered.add(k[0])
                        covered.add(k[1])
                pairs_to_merge = pairs_to_merge[: max(1, len(R.partition) // 8)]

                if pairs_to_merge:
                    """********* merge *********"""
                    print_(f"******* MERGING {len(pairs_to_merge)} pair(s) *******")
                    for p in pairs_to_merge:
                        print_(
                            f"  merging subspaces {p[0]} (size {R.partition[p[0]]}) and {p[1]} (size {R.partition[p[1]]}) - MI: {metric[p]:.4f} > threshold {cfg.merge_thr}"
                        )

                    temp = [j for p in pairs_to_merge for j in p]
                    clusters = pairs_to_merge.copy()
                    for j in range(len(R.partition)):
                        if j not in temp:
                            clusters.append((j,))
                    clusters_sizes = []
                    for c in clusters:
                        clusters_sizes.append((c, sum(R.partition[j] for j in c)))
                    clusters_sizes.sort(key=lambda x: -x[1])

                    R_chunks = R.R.weight.data.split(R.partition, dim=1)
                    new_R = []
                    new_partition = []
                    for c, s in clusters_sizes:
                        new_R.extend([R_chunks[j] for j in c])
                        new_partition.append(s)
                    new_R = torch.cat(new_R, dim=1)

                    R = NewUnevenRTrainer(
                        h_dim, new_partition, cfg, buffer, previous_R=new_R
                    ).to(cfg.device)
                    assert torch.allclose(R.R.weight.data, new_R), (
                        (R.R.weight.data - new_R).abs().mean().item()
                    )
                    optimizer = torch.optim.Adam(
                        R.parameters(),
                        lr=cfg.lr,
                        betas=(cfg.adam_beta1, cfg.adam_beta2),
                    )

                    print_(
                        f"******* after merging: {len(clusters_sizes)} subspaces with sizes {[s for _, s in clusters_sizes]}"
                    )

                else:
                    max_mi = lis[0][1] if lis else 0.0
                    print_(
                        f"******* NO MERGE: highest MI ({max_mi:.4f}) is below threshold ({cfg.merge_thr})"
                    )
                    if len(R.partition) <= 2:
                        print_(
                            f"******* STOPPING: only {len(R.partition)} subspace(s) remain and no pairs qualify for merging"
                        )
                    break

        # Determine why training finished
        if i + 1 >= cfg.max_steps:
            print_(f"******* FINISHED: reached max_steps ({cfg.max_steps})")
        else:
            print_(
                f"******* FINISHED early at step {i+1}: no more merges possible (all MI values below threshold {cfg.merge_thr})"
            )
        R.save(output_dir, suffix=f"-{model_name}-{site_name}")

        print_(f"evaluating ({test_search_steps} steps)...")
        eval_result = []
        for j in range(max(1, 100 * 128 // cfg.test_batch_size)):
            eval_result.append(
                R.evaluate_step(
                    num_steps=test_search_steps, batch_size=cfg.test_batch_size
                )
            )
        eval_result = torch.stack(eval_result).mean(dim=0)
        print_(f" ******* eval result *******")
        print_(
            "mean (weighted)",
            (eval_result * torch.tensor(R.partition, device=device)).sum().item()
            / sum(R.partition),
        )
        print_("mean (unweighted)", eval_result.mean().item())
        print_(eval_result)

        f.close()
