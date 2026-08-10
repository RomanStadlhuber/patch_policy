# Patch Policy: Efficient Embodied Control via Dense Visual Representations

[[Project Website]](https://patch-policy.github.io/) [[Paper]](https://arxiv.org/abs/2607.18236v1) [[Dataset]](https://huggingface.co/datasets/gaoyuezhou/patch-policy-datasets)

[Gaoyue Zhou](https://gaoyuezhou.github.io/), [Zichen Jeff Cui](https://jeffcui.com/), [Ada Langford](https://www.linkedin.com/in/ada-langford-231883332/), [Bowen Tan](https://bowen-tan.com/), [Yann LeCun](http://yann.lecun.com/) and [Lerrel Pinto](https://www.lerrelpinto.com/), New York University, Meta AI, AMI Labs

https://github.com/user-attachments/assets/3de8fc0d-9411-41f3-84c3-9e79b899e144

![Method overview](assets/method.png)

This repo contains code for training and reproducing sim environment experiments across four simulation environments: Push-T, Block Pushing, LIBERO Goal, and Cube.

## Getting Started

1. [Setup](#setup)
2. [Datasets](#datasets)
3. [Training a policy](#training-a-policy)
4. [Choosing the visual encoder](#choosing-the-visual-encoder)

## Setup

Create the conda environment (this installs everything, including the CUDA build of PyTorch):

```
conda env create -f conda_env.yml
conda activate patch-policy
```

Tested on Ubuntu 22.04 with CUDA 12.8. To log training runs, log in to Weights & Biases with `wandb login` (or set `export WANDB_MODE=disabled` to turn logging off). In `./configs/env_vars/env_vars.yaml`, set `wandb_entity` to your wandb username.

## Datasets

The datasets for all four simulation environments are hosted on the Hugging Face Hub:
[**gaoyuezhou/patch-policy-datasets**](https://huggingface.co/datasets/gaoyuezhou/patch-policy-datasets).

| Environment    | Folder (after unzip) | Zip size |
| -------------- | -------------------- | -------- |
| Push-T         | `pusht_dataset`      | 61 MB    |
| Cube           | `cube_dataset`       | 2.2 GB   |
| LIBERO Goal    | `libero_dataset`     | 7.7 GB   |
| Block Pushing  | `block_push_dataset` | 5.9 GB   |

### Download

1. Install the Hugging Face Hub CLI:
   ```
   pip install "huggingface_hub==0.36.2"
   ```
2. Download the dataset repo to a local directory (this is the directory all four datasets will live in):
   ```
   huggingface-cli download gaoyuezhou/patch-policy-datasets \
     --repo-type dataset --local-dir patch_policy_datasets
   ```
   To download only a subset, add e.g. `--include "pusht_dataset.zip"`.
3. Unzip each dataset in place:
   ```
   cd patch_policy_datasets
   for f in *.zip; do unzip -q "$f"; done
   ```

### Point the code at the data

The dataloading code reads from a single `dataset_root` directory — no code changes are needed, you only set this path.

- In `./configs/env_vars/env_vars.yaml`, set `dataset_root` to the unzipped directory (e.g. the absolute path to `patch_policy_datasets`), and set `save_path` to where you want training/rollout results saved (e.g. the root directory of this repo).

The expected layout under `dataset_root` is:
```
patch_policy_datasets/
├── pusht_dataset/
├── cube_dataset/
├── libero_dataset/
└── block_push_dataset/
```

> **Note:** The `.pth`/`.npy`/`.pkl` files are loaded with `torch.load` / `numpy`, which unpickle data. Only use datasets you trust.

## Training a policy

Policy training and online evaluation both run through `train_policy.py`, driven by the configs in `configs/`. A run trains the policy on top of a **frozen visual encoder** and periodically rolls it out in the simulator.

```
python train_policy.py --config-name train_pusht        # Push-T
python train_policy.py --config-name train_blockpush    # Block Pushing
python train_policy.py --config-name train_cube         # Cube
MUJOCO_GL=egl python train_policy.py --config-name train_libero_goal   # LIBERO Goal
```

- **Diffusion policy** variants are available for every environment — append `_diffusion` to the config name (e.g. `train_pusht_diffusion`). The default configs use a VQ-BeT policy head.
- Checkpoints are written under `save_path`.

### Single-GPU configs

The config names above assume a node of 8 GPUs. We also provide `_1gpu` variants of
every config, tuned to fit within 32 GB of VRAM:

```
python train_policy.py --config-name train_pusht_1gpu
python train_policy.py --config-name train_blockpush_1gpu
python train_policy.py --config-name train_cube_1gpu
MUJOCO_GL=egl python train_policy.py --config-name train_libero_goal_1gpu
```

These runs use DINOv2 ViT-S (`dino_patch`) and precompute the frozen encoder's
features once at startup. Observation windows are shortened where memory requires
it — LIBERO Goal VQ-BeT drops from 10 to 2, and Cube from 5 to 2 — and batch sizes
are set per environment. `_diffusion` variants are available here too (e.g.
`train_pusht_diffusion_1gpu`).

Launcher configs for submitting to a SLURM cluster are in `configs/cluster/`.

### Choosing the visual encoder

The encoder is frozen and selected via the `encoder` config group (`configs/encoder/`); off-the-shelf encoders need no training. The default is DINOv2 patch features (`dino_patch`). Override it on the command line:

```
python train_policy.py --config-name train_pusht encoder=webssl_patch
python train_policy.py --config-name train_pusht encoder=vjepa2_patch
python train_policy.py --config-name train_pusht encoder=dinov3_patch
python train_policy.py --config-name train_pusht encoder=siglip2_patch
```

Each of these uses dense patch features. Two pooled variants are also provided for comparison: `*_patch_avg_pool` (patch tokens mean-pooled into a single vector) is available for every encoder above as well as `dino`, and `*_cls` (the CLS token instead of patch tokens) is available for `dino`, `dinov3`, and `webssl`. ResNet-18 baselines (`resnet18_imagenet`, `resnet18_random`) and pretrained [DynaMo](https://dynamo-ssl.github.io/) encoders are also supported — see `configs/encoder/` for the full list.

> **Note (DINOv3):** the DINOv3 weights are hosted in a gated Hugging Face repo. To use the `dinov3_*` encoders, request access at [facebook/dinov3-vits16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) and log in with `hf auth login` before launching.

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{zhou2026patchpolicyefficientembodied,
      title={Patch Policy: Efficient Embodied Control via Dense Visual Representations}, 
      author={Gaoyue Zhou and Zichen Jeff Cui and Ada Langford and Bowen Tan and Yann LeCun and Lerrel Pinto},
      year={2026},
      eprint={2607.18236},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.18236}, 
}
```
