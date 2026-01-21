from transformer_lens import HookedTransformer
from datasets import load_dataset, concatenate_datasets, load_from_disk
import torch
from tqdm import tqdm
import gc
import random
import re
from collections import deque
import torch.nn.functional as F


class BufferReuse:
    # a buffer designed for extended search range
    def __init__(self, cfg, model: HookedTransformer, normalize=False):
        self.cfg = cfg

        self.load_dataset()

        self.block_len = cfg.block_len
        self.refresh_block_num = cfg.refresh_block_num

        self.blocks = deque()
        self.cursor = 0

        self.cfg = cfg
        self.model = model
        self.buffer_dtype = torch.float16
        assert not normalize

        self.refresh()

        self.mean = self.compute_mean()

        self.to_pop = None

    def load_dataset(self):
        self.data = iter(self.cfg.dataset)

    @torch.no_grad()
    def compute_mean(self):
        s = 0
        for block in self.blocks:
            s += block.float().mean(
                dim=0, keepdim=True
            )  # important: first float() then mean()
        return s / len(self.blocks)

    def __iter__(self):
        return self

    def pop_one(self, bz=None):
        if len(self.blocks) == 0:
            self.refresh()

        self.cursor = 0
        if bz is None or bz == self.block_len:
            acts = self.blocks.popleft()
        elif bz > self.block_len:
            n = -(bz // -self.block_len)
            acts = torch.cat([self.blocks.popleft() for _ in range(n)], dim=0)
        elif self.to_pop is None or self.to_pop.size(0) < bz:
            acts = self.blocks.popleft()
            acts, self.to_pop = acts.split([bz, acts.size(0) - bz], dim=0)
        else:
            acts, self.to_pop = self.to_pop.split([bz, self.to_pop.size(0) - bz], dim=0)

        return acts.contiguous()

    def __next__(self):
        """
        Return a batch of activations
        """
        with torch.no_grad():
            if self.cursor >= len(self.blocks):
                self.refresh()

            out = self.blocks[self.cursor]
            self.cursor += 1

            return out

    def token_batch(self):
        """
        Return a batch of tokens (flattened) and corresponding mask.

        Returns:
            tuple: (tokens, mask) where mask is a list of 1s and 0s indicating
                   which tokens should be included (1) or excluded (0).
                   mask is None if no masking is needed.
        """
        try:
            tokens = []
            mask = []
            has_mask = False
            while True:
                sample = next(self.data)
                # Handle dict samples from masked datasets
                if isinstance(sample, dict):
                    text = sample["text"]
                    sample_mask = sample.get("mask")
                    if sample_mask is not None:
                        has_mask = True
                else:
                    text = sample
                    sample_mask = None

                input_ids = self.model.tokenizer(text)["input_ids"]
                tokens.extend(input_ids)

                # Build mask for these tokens
                if sample_mask is not None:
                    mask.extend(sample_mask)
                else:
                    mask.extend([1] * len(input_ids))

                if len(tokens) >= self.cfg.caching_batch_size * self.model.cfg.n_ctx:
                    break
                tokens.append(self.model.tokenizer.eos_token_id)
                mask.append(1)  # EOS tokens are always included

            tokens = tokens[: self.cfg.caching_batch_size * self.model.cfg.n_ctx]
            mask = mask[: self.cfg.caching_batch_size * self.model.cfg.n_ctx]
            return tokens, mask if has_mask else None

        except StopIteration:
            print("End of data stream reached")
            self.load_dataset()
            return self.token_batch()

    @torch.no_grad()
    def refresh(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.cfg.act_site != "blocks.0.hook_resid_pre":
            self.model.reset_hooks()
            cache = self.model.add_caching_hooks(self.cfg.act_site)
            stop_at_layer = (
                int(re.search(r"blocks\.(\d+)\.", self.cfg.act_site).group(1)) + 1
            )
        else:
            stop_at_layer = 0

        buffer = []
        buffer_size = 0  # Track total number of activations

        pbar = tqdm(
            total=self.block_len * self.refresh_block_num,
            desc="Refreshing activations",
            disable=True,
        )
        while buffer_size < self.block_len * self.refresh_block_num:
            # inside no_grad()
            input_batch, batch_mask = self.token_batch()
            input_batch = torch.tensor(
                input_batch, device=self.cfg.device, dtype=torch.long
            ).view(self.cfg.caching_batch_size, -1)

            # Use autocast for the appropriate device type
            device_type = "cuda" if self.cfg.device.type == "cuda" else "cpu"
            with torch.autocast(device_type, enabled=(device_type == "cuda")):
                acts = self.model(input_batch, stop_at_layer=stop_at_layer)
                if self.cfg.act_site != "blocks.0.hook_resid_pre":
                    acts = cache[self.cfg.act_site]
                # acts = F.layer_norm(acts, [acts.size(-1)])

            acts = acts.flatten(end_dim=1).to(self.buffer_dtype)

            # Apply mask to filter out masked token activations
            if batch_mask is not None:
                mask_tensor = torch.tensor(
                    batch_mask, device=acts.device, dtype=torch.bool
                )
                acts = acts[mask_tensor]

            buffer.append(acts)
            buffer_size += acts.size(0)
            pbar.update(acts.size(0))

        pbar.close()

        # Concatenate all activations and shuffle using indices (faster than shuffling list of tensors)
        buffer = torch.cat(buffer, dim=0)
        perm = torch.randperm(buffer.size(0))
        buffer = buffer[perm]

        # Split into blocks
        for i in range(self.refresh_block_num):
            start_idx = i * self.block_len
            end_idx = start_idx + self.block_len
            self.blocks.append(buffer[start_idx:end_idx].contiguous())

        assert self.blocks[-1].dtype == self.buffer_dtype
