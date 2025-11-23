from dataclasses import dataclass
from subspace_partition.training.data import *
from subspace_partition.training.model import *
from subspace_partition.training.utils import *
from functools import partial
from matplotlib import pyplot as plt
from collections import defaultdict
from transformer_lens import HookedTransformerConfig


@dataclass
class SubspacePartitionConfig:
    """Configuration class for configuring the subspace partitioning method.

    Args:
        exp_name: Name of the experiment.
        batch_size: Batch size for query.
        test_batch_size: Batch size for testing (128*512/block_len when unit_size=4).
        acc_steps: Number of accumulation steps.
        metric: Metric to use (e.g., "euclidean").
        max_steps: Maximum number of training steps for R.
        merge_interval: Interval for merging.
        merge_start: Step to start merging.
        merge_thr: Threshold for merging.
        merge_metric: Metric to use for merging (e.g., "mi").
        search_steps: Number of search steps.
        unit_size: Size of the unit.
        model_name: Name of the model (e.g., "gpt2").
        lr: Learning rate.
        adam_beta1: Beta1 parameter for Adam optimizer.
        adam_beta2: Beta2 parameter for Adam optimizer.
        weight_type: Type of weight (e.g., "none").
        block_len: Length of the block.
        clip_grad: Gradient clipping value.
        data_source: Source of the data (e.g., "minipile", "openwebtext").
        double_q: Whether to use double Q-learning.
    """

    exp_name: str
    act_sites: list[str]
    model_name: str | None = None
    model_config: HookedTransformerConfig | None = None
    model_weights_path: Path | None = None
    batch_size: int = 128  # for query
    test_batch_size: int = 128  # 128*512/block_len when unit_size=4
    acc_steps: int = 1
    metric: str = "euclidean"
    max_steps: int = 50_000  # for training R
    merge_interval: int = 3_000
    merge_start: int = 10_000
    merge_thr: float = 0.04
    merge_metric: str = "mi"
    search_steps: int = 25
    unit_size: int = 32
    lr: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_type: str = "none"
    block_len: int = 16384
    clip_grad: float = 100.0
    data_source: str = "minipile"  # minipile, openwebtext
    double_q: bool = False
    device: torch.device | None = None
    output_dir: Path | None = None
    act_site: str | None = None  # don't set this manually

    def __post_init__(self):
        assert self.model_name is not None or (
            self.model_config is not None and self.model_weights_path is not None
        ), "Either model_name or both model_config and model_weights_path must be provided."

        self.refresh_block_num: int = 2048 * 2048 // self.block_len
        self.caching_batch_size: int = (
            16 if self.model_name != "gemma" else 2
        )  # because 8192 n_ctx

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        if self.model_config is not None:
            model_config_dict = self.model_config.to_dict()
            # Convert device in model_config to string for JSON serialization
            if "device" in model_config_dict:
                model_config_dict["device"] = str(model_config_dict["device"])
            d["model_config"] = model_config_dict
        if self.device is not None:
            d["device"] = str(self.device)
        if self.model_weights_path is not None:
            d["model_weights_path"] = str(self.model_weights_path)
        if self.output_dir is not None:
            d["output_dir"] = str(self.output_dir)
        return d

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> "SubspacePartitionConfig":
        return cls(**cfg_dict)


def run_subspace_partition(cfg: SubspacePartitionConfig):
    set_seed(0)

    if cfg.output_dir is not None:
        output_dir: Path = cfg.output_dir / cfg.exp_name
    else:
        output_dir = Path(cfg.exp_name)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if any(output_dir.iterdir()):
        raise ValueError(f"Output directory {output_dir} is not empty.")

    config_path = output_dir / "training_args.json"
    with open(config_path, "w") as f:
        ...
        # json.dump(cfg.to_dict(), f)

    test_search_steps = 200 * 2048 // cfg.block_len
    if cfg.unit_size <= 4:
        mi_search_steps = 5 * 2048 // cfg.block_len
    else:
        mi_search_steps = 50 * 2048 // cfg.block_len

    device = cfg.device

    if cfg.model_name is not None:
        hooked_model = HookedTransformer.from_pretrained(
            to_valid_model_name(cfg.model_name),
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            device=device,
        )
    else:
        hooked_model = HookedTransformer(cfg.model_config).to(device)
        state_dict = torch.load(cfg.model_weights_path, map_location=device)
        hooked_model.load_state_dict(state_dict)

    h_dim = hooked_model.cfg.d_model

    for act_site in cfg.act_sites:
        print("training for", act_site)
        cfg.act_site = act_site
        site_name = site_name_to_short_name(act_site)

        if (output_dir / f"R-{cfg.model_name}-{site_name}.pt").exists():
            continue
        log_path = output_dir / f"train_log-{cfg.model_name}-{site_name}.txt"
        f = open(log_path, "w")
        print_ = partial(print_to_both, f=f)

        if not cfg.double_q:
            buffer = BufferReuse(cfg, hooked_model)
        else:
            buffer = BufferReuseDoubleQueue(cfg.to_dict(), hooked_model)
        R = NewUnevenRTrainer(
            h_dim, [cfg.unit_size] * (h_dim // cfg.unit_size), cfg.to_dict(), buffer
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
                        h_dim, new_partition, cfg.to_dict(), buffer, previous_R=new_R
                    ).to(cfg.device)
                    assert torch.allclose(R.R.weight.data, new_R), (
                        (R.R.weight.data - new_R).abs().mean().item()
                    )
                    optimizer = torch.optim.Adam(
                        R.parameters(),
                        lr=cfg.lr,
                        betas=(cfg.adam_beta1, cfg.adam_beta2),
                    )

                    print_(f"******* after merging ({cfg.merge_thr}):", clusters_sizes)

                else:
                    break

        print_(f"finish training ({i+1})")
        R.save(output_dir, suffix=f"-{cfg.model_name}-{site_name}")

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
