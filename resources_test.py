import einops
import os
import random
import time
from collections import deque

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from envs.venv import SubprocVectorEnv
from utils.normalizer import LinearNormalizer

OmegaConf.register_new_resolver("eval", eval, replace=True)

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_params(num_params):
    """Format parameter count in human-readable format."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


@hydra.main(config_path="eval_configs", version_base="1.2")
def main(cfg):
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    seed_everything(cfg.seed)

    use_diffusion = "diffusion" in cfg.model['_target_']

    # Initialize encoder
    encoder = hydra.utils.instantiate(cfg.encoder)
    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad = False

    if cfg.load_path:
        print(f"Loading model from {cfg.load_path} ...")
        cbet_model = torch.load(
            cfg.load_path, map_location=device, weights_only=False
        ).to(device)
    else:
        cbet_model = hydra.utils.instantiate(cfg.model).to(device)

    # Initialize dataset for normalizer (if diffusion)
    if use_diffusion and getattr(cbet_model, "normalizer", None) is None:
        dataset = hydra.utils.instantiate(cfg.dataset)
        actions = dataset.get_all_actions()
        action_normalizer = LinearNormalizer()
        action_normalizer.fit(actions)
        cbet_model.set_normalizer(action_normalizer)

    cbet_model.eval()

    # ==========================================
    # 1) & 2) Print parameter counts
    # ==========================================
    print("\n" + "=" * 60)
    print("MODEL PARAMETER COUNTS")
    print("=" * 60)

    # Encoder parameters
    enc_total, enc_trainable = count_parameters(encoder)
    print(f"\nEncoder:")
    print(f"  Total parameters:     {format_params(enc_total):>10} ({enc_total:,})")
    print(f"  Trainable parameters: {format_params(enc_trainable):>10} ({enc_trainable:,})")

    # Policy model parameters
    model_total, model_trainable = count_parameters(cbet_model)
    print(f"\nPolicy Model (cbet_model):")
    print(f"  Total parameters:     {format_params(model_total):>10} ({model_total:,})")
    print(f"  Trainable parameters: {format_params(model_trainable):>10} ({model_trainable:,})")

    # Combined
    combined_total = enc_total + model_total
    combined_trainable = enc_trainable + model_trainable
    print(f"\nCombined (Encoder + Policy):")
    print(f"  Total parameters:     {format_params(combined_total):>10} ({combined_total:,})")
    print(f"  Trainable parameters: {format_params(combined_trainable):>10} ({combined_trainable:,})")

    # ==========================================
    # 3) Measure inference speed
    # ==========================================
    print("\n" + "=" * 60)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 60)

    # Create environment and get one observation
    env_fn = lambda: hydra.utils.instantiate(cfg.env.gym)
    env = SubprocVectorEnv([env_fn for _ in range(1)])  # Single env
    env.seed([cfg.seed])

    use_libero_goal = cfg.data.get("use_libero_goal", False)

    # Get initial observation
    obs = env.reset(goal_idx=0)  # N V C H W, where N=1
    assert obs.min() >= 0 and obs.max() <= 1, "expect 0-1 range observation"

    # Embed observation
    def embed(enc, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)  # N V C H W
        return einops.rearrange(enc(obs), "N V P E -> N (V P) E")

    with torch.no_grad():
        obs_enc = embed(encoder, obs)  # N (V P) E

    # Create observation stack
    obs_stack = deque(maxlen=cfg.eval_window_size)
    for _ in range(cfg.eval_window_size):
        obs_stack.append(obs_enc)

    # Prepare inputs
    obs_tensor = torch.stack(tuple(obs_stack)).float().to(device)
    obs_tensor = einops.rearrange(obs_tensor, "T N P E -> N T P E")

    # Create dummy goal
    if use_libero_goal:
        # Use the same observation as goal for benchmarking
        goal = obs_enc.clone()
    else:
        goal = torch.zeros(1, device=device)

    goal = torch.as_tensor(goal, dtype=torch.float32, device=device)
    if goal.ndim > 1:
        goal = einops.repeat(goal, '... -> N T ...', N=1, T=cfg.eval_window_size)
    else:
        goal = einops.repeat(goal, '... -> N T ...', N=1, T=cfg.eval_window_size)

    # Warmup runs
    num_warmup = 50
    num_benchmark = cfg.get("num_benchmark_iters", 50)

    print(f"\nWarming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = cbet_model(obs_tensor, goal, None)
            torch.cuda.synchronize() if device.type == 'cuda' else None

    # Benchmark inference speed (model only, no encoder)
    print(f"Benchmarking policy model inference ({num_benchmark} iterations)...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_benchmark):
            _ = cbet_model(obs_tensor, goal, None)
            torch.cuda.synchronize() if device.type == 'cuda' else None

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    policy_freq = num_benchmark / elapsed

    print(f"\nPolicy Model Only:")
    print(f"  Total time:       {elapsed:.3f} seconds")
    print(f"  Iterations:       {num_benchmark}")
    print(f"  Control frequency: {policy_freq:.2f} Hz")
    print(f"  Latency per call:  {1000 / policy_freq:.3f} ms")

    # Benchmark full pipeline (encoder + model)
    print(f"\nBenchmarking full pipeline (encoder + policy) ({num_benchmark} iterations)...")
    obs_raw = torch.as_tensor(obs, dtype=torch.float32, device=device)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_benchmark):
            # Encode observation
            obs_enc_bench = einops.rearrange(encoder(obs_raw), "N V P E -> N (V P) E")
            # Stack (simulate window)
            obs_stacked = einops.repeat(obs_enc_bench, 'N P E -> N T P E', T=cfg.eval_window_size)
            # Run policy
            _ = cbet_model(obs_stacked, goal, None)
            torch.cuda.synchronize() if device.type == 'cuda' else None

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    full_freq = num_benchmark / elapsed

    print(f"\nFull Pipeline (Encoder + Policy):")
    print(f"  Total time:       {elapsed:.3f} seconds")
    print(f"  Iterations:       {num_benchmark}")
    print(f"  Control frequency: {full_freq:.2f} Hz")
    print(f"  Latency per call:  {1000 / full_freq:.3f} ms")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Parameters:      {format_params(combined_total)}")
    print(f"Trainable Parameters:  {format_params(combined_trainable)}")
    print(f"Policy-only Freq:      {policy_freq:.2f} Hz")
    print(f"Full Pipeline Freq:    {full_freq:.2f} Hz")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
