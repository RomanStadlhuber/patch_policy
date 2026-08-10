import torch


# for supporting output_dim as metadata for the config
def from_ckpt(f: str, output_dim: int, n_patches: int):
    model = torch.load(f, weights_only=False)
    return model