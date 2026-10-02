"""Roll out a trained wedge diffusion policy in Drake with a Meshcat 3rd-person view.

    python -m rollouts.wedge_meshcat outputs/wedge/2026-10-02/13-12-02 --episodes 5 --host 100.94.69.64

Loads the run's config (`.hydra/config.yaml`) and its best checkpoint (the `model_<epoch>.pt` with
the highest epoch; only new best evals write one), then plays eval set entries 0 .. N-1 one after
another in this process. Each successful episode is recorded and written to
`<out_dir>/entry_XXX.html` (every episode with `--save-all`), a standalone Meshcat page whose
animation slider scrubs the rollout. The live Meshcat server stays up after the last episode until
Ctrl-C.
"""

import argparse
import re
import time
from collections import deque
from pathlib import Path

import einops
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


def best_checkpoint(run_dir: Path) -> Path:
    """The `model_<epoch>.pt` with the highest epoch, i.e. the last new best eval."""
    ckpts = [(int(m.group(1)), p) for p in run_dir.glob("model_*.pt") if (m := re.fullmatch(r"model_(\d+)\.pt", p.name))]
    if not ckpts:
        raise FileNotFoundError(f"no model_<epoch>.pt in {run_dir}")
    return max(ckpts)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Training run folder with .hydra/config.yaml and model_*.pt")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path; default: the run's best model_<epoch>.pt")
    parser.add_argument("--episodes", type=int, default=5, help="Eval set entries 0 .. N-1")
    parser.add_argument("--host", default="localhost", help="Meshcat bind address, e.g. the Tailscale IP")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--inference_steps", type=int, default=None, help="DDIM steps; default: final_inference_steps")
    parser.add_argument("--out_dir", default=None, help="Recordings folder; default: <run_dir>/meshcat")
    parser.add_argument("--save-all", action="store_true", help="Write every episode's recording, not only successes")
    parser.add_argument("--pause", action="store_true", help="Wait for Enter after each episode to scrub it live")
    args = parser.parse_args()

    # imported here: pydrake is a wedge-only dependency
    from pydrake.geometry import Meshcat, MeshcatParams
    from supermanipulation.wedge.actions import decode_chunk
    from supermanipulation.wedge.env import WedgeGraspEnv, stack_frames, stack_state

    run_dir = Path(args.run_dir).resolve()
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    ckpt = Path(args.ckpt) if args.ckpt else best_checkpoint(run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "meshcat"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device

    encoder = hydra.utils.instantiate(cfg.encoder).to(device).eval()
    print(f"Loading model from {ckpt} ...")
    # the file holds a whole nn.Module, not tensors
    model = torch.load(ckpt, map_location=device, weights_only=False).to(device)
    model.train(False)
    model.set_inference_steps(args.inference_steps or cfg.get("final_inference_steps", 100))

    meshcat = Meshcat(MeshcatParams(host=args.host, port=args.port))
    print(f"Meshcat: {meshcat.web_url()}")
    env = WedgeGraspEnv(eval_set=cfg.env.gym.eval_set, meshcat=meshcat)
    env.seed(cfg.seed)

    def embed(obs):
        frames = torch.as_tensor(stack_frames(obs), dtype=torch.float32, device=device)  # 1 V C H W
        return einops.rearrange(encoder(frames), "N V P E -> N (V P) E")

    # mirrors train_policy.py's rollout `goal_fn` and the training goals from `WedgeDataset.get_frames`
    # (datasets/wedge.py); update this together with them when goal conditioning changes
    goal = torch.zeros(1, device=device)  # wedge has no goal input (goal_dim 0)
    results = []
    for episode in range(args.episodes):
        raw_obs = env.reset()
        obs_stack = deque([embed(raw_obs)], maxlen=cfg.eval_window_size)
        done, start = False, time.perf_counter()
        with torch.no_grad():
            while not done:
                obs = einops.rearrange(torch.stack(tuple(obs_stack)), "T N P E -> N T P E")
                goals = einops.repeat(goal, "... -> N T ...", N=1, T=cfg.eval_window_size)
                action, _, _ = model(obs, goals, None)
                # heel-frame deltas from the measured state at t to absolute targets in T
                actions = decode_chunk(action.cpu().numpy(), *stack_state(raw_obs))
                for t in range(actions.shape[1]):
                    raw_obs, _, done, info = env.step(actions[0, t])
                    obs_stack.append(embed(raw_obs))
                    if done:
                        break
        env.publish_recording()
        saved = ""
        if args.save_all or info["success"]:
            html = out_dir / f"entry_{info['entry']:03d}.html"
            html.write_text(meshcat.StaticHtml())
            saved = f" -> {html}"
        results.append(info)
        print(
            f"entry {info['entry']:3d}: {info['phase']:8s} {info.get('fail_reason') or '':13s} p_score {info['p_score']:.3f} "
            f"steps {info['steps']:3d} ({time.perf_counter() - start:.0f} s){saved}"
        )
        if args.pause and episode < args.episodes - 1:
            input("Scrub the recording in Meshcat, Enter for the next episode ... ")

    success = np.mean([r["success"] for r in results])
    p_score = np.mean([r["p_score"] for r in results])
    print(f"{len(results)} episodes: success {success:.0%}, p_score mean {p_score:.3f}")
    print("Last recording stays live in Meshcat. Ctrl-C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
