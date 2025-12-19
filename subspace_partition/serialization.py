import json
import torch
import transformer_lens


class HookedTransformerConfigEncoder(json.JSONEncoder):
    """Custom JSON encoder for HookedTransformerConfig to handle torch.device and torch.dtype."""

    def default(self, obj):
        if isinstance(obj, torch.device):
            return {"__torch_device__": str(obj)}
        elif isinstance(obj, torch.dtype):
            return {"__torch_dtype__": str(obj).split(".")[-1]}
        return super().default(obj)


def hooked_transformer_config_decoder(dct):
    """Custom JSON decoder for HookedTransformerConfig to handle torch.device and torch.dtype."""

    if "__torch_device__" in dct:
        return torch.device(dct["__torch_device__"])
    elif "__torch_dtype__" in dct:
        return getattr(torch, dct["__torch_dtype__"])
    return dct
