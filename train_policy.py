import einops
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

import wandb
from utils.video import VideoRecorder
from utils.metrics_logger import MetricsLogger
import pickle
from datasets.core import (
    FileEmbeddingDataset,
    TrajectoryEmbeddingDataset,
    TrajectorySlicerDataset,
    embedding_cache_key,
    split_traj_datasets,
)
from envs.venv import SubprocVectorEnv
from utils.normalizer import LinearNormalizer
from datetime import timedelta
from accelerate import Accelerator
import logging
from accelerate import InitProcessGroupKwargs, DistributedDataParallelKwargs
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
import torch.distributed as dist

from resources_test import format_params, count_parameters

OmegaConf.register_new_resolver("eval", eval, replace=True)

os.environ["WANDB_START_METHOD"] = "thread"
logger = logging.getLogger(__name__)

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


def save_snapshot(save_path, cbet_model, optimizer, accelerator, epoch,
                  wandb_run_id, metrics_history, reward_history, best_eval_metric):
    # Snapshot the full training state so a preempted/requeued job can resume.
    cbet_model_raw = accelerator.unwrap_model(cbet_model)
    snapshot = {
        "model": cbet_model_raw,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "wandb_run_id": wandb_run_id,
        "metrics_history": metrics_history,
        "reward_history": reward_history,
        "best_eval_metric": best_eval_metric,
    }
    # Atomic write: write to a temp file then rename, so a preemption mid-save
    # never leaves a corrupt snapshot.pt behind.
    tmp_path = save_path / "snapshot.pt.tmp"
    torch.save(snapshot, tmp_path)
    tmp_path.rename(save_path / "snapshot.pt")


def load_snapshot(snapshot_path, cbet_model, optimizer, accelerator):
    snapshot = torch.load(
        snapshot_path, map_location=accelerator.device, weights_only=False
    )
    # Load the full model object, then copy its state into the prepared model
    saved_model = snapshot["model"]
    cbet_model_raw = accelerator.unwrap_model(cbet_model)
    # DiffusionPolicy assigns ema_noise_pred_net (an alias of
    # ema.averaged_model) lazily at first rollout, so a snapshot taken after
    # a rollout has state_dict keys a freshly built model lacks. Mirror the
    # lazy init before loading so the key sets match.
    if (
        getattr(saved_model, "ema_noise_pred_net", None) is not None
        and getattr(cbet_model_raw, "ema_noise_pred_net", None) is None
    ):
        cbet_model_raw.ema_noise_pred_net = cbet_model_raw.ema.averaged_model
    cbet_model_raw.load_state_dict(saved_model.state_dict())
    # vqvae_is_fit is a plain attribute, not part of state_dict. Without it a
    # resumed VQ-BeT re-fits a new codebook under the trained heads and
    # silently destroys the policy.
    if hasattr(saved_model, "vqvae_is_fit"):
        cbet_model_raw.vqvae_is_fit = saved_model.vqvae_is_fit
    optimizer.load_state_dict(snapshot["optimizer_state_dict"])
    return snapshot


@hydra.main(config_path="configs", version_base="1.2")
def main(cfg):
    if HydraConfig.get().mode == RunMode.MULTIRUN:
        logger.info(" Multirun setup begin...")
        logger.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
        logger.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
        # ==== init ddp process group ====
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(hours=2),  # Extended timeout for eval variance
            )
            logger.info("Multirun setup completed.")
        except Exception as e:
            logger.error(f"DDP setup failed: {e}")
            raise
        torch.distributed.barrier()
    process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(hours=2),  # Extended timeout for eval variance
        )
    dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="wandb", kwargs_handlers=[process_group_kwargs, dist_kwargs]
    )
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    logger.info(
        f"rank: {accelerator.local_process_index}"
    )
    device = accelerator.device
    logger.info(f"device: {device}")
    assert cfg.batch_size % accelerator.num_processes == 0, (
        "Batch size must be divisible by the number of processes. "
        f"Batch_size: {cfg.batch_size} num_processes: {accelerator.num_processes}."
    )

    OmegaConf.set_struct(cfg, False)
    cfg.effective_batch_size = cfg.batch_size
    cfg.gpu_batch_size = cfg.batch_size // accelerator.num_processes
    OmegaConf.set_struct(cfg, True)

    accelerator.wait_for_everyone()

    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    use_diffusion = "diffusion" in cfg.model['_target_']

    save_path = Path(os.getcwd())
    model_name = os.getcwd().split("outputs/")[-1]
    slurm_job_id = os.environ.get("SLURM_JOB_ID")  # logged to wandb below
    logger.info(f"Run folder (save_path)={save_path}")

    # Peek at the snapshot for the wandb run id before initializing wandb, so a
    # resumed job continues logging to the same run instead of starting a new one.
    snapshot_path = save_path / "snapshot.pt"
    wandb_run_id = None
    if snapshot_path.exists():
        peek = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        wandb_run_id = peek.get("wandb_run_id")
        del peek

    # Hydra changes working directory to the run directory, so just use current directory
    if accelerator.is_main_process:
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_config["slurm_job_id"] = slurm_job_id
        wandb_kwargs = dict(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_config,
            name=model_name,  # Use current directory name
        )
        if wandb_run_id is not None:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "must"
        run = wandb.init(**wandb_kwargs)
        wandb_run_id = run.id

    print("Saving to {}".format(os.getcwd()))
    video = VideoRecorder(dir_name=save_path)

    # init datasets
    dataset = hydra.utils.instantiate(cfg.dataset)
    encoder = hydra.utils.instantiate(cfg.encoder)
    # encoder = encoder.to(cfg.device).eval()
    encoder = accelerator.prepare(encoder)

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    # save_model() writes the whole model as one pickled module, so a
    # checkpoint replaces cbet_model outright and has to be loaded before the
    # setup, or else the normalizer and the optimizer bind to the model object we then discard.
    if cfg.load_path:
        ckpt = Path(cfg.load_path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"load_path must be a model_<tag>.pt file written by save_model(), "
                f"got: {ckpt}"
            )
        print(f"Loading model from {ckpt} ...")
        # the file holds a whole nn.Module, not tensors
        cbet_model = torch.load(
            ckpt, map_location=cfg.device, weights_only=False
        ).to(cfg.device)
    else:
        cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)

    if use_diffusion:
        if getattr(cbet_model, "normalizer", None) is not None:
            print("Using the action normalizer stored in the checkpoint")
        else:
            actions = dataset.get_all_actions()
            action_normalizer = LinearNormalizer()
            action_normalizer.fit(actions)
            cbet_model.set_normalizer(action_normalizer)

    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )

    cbet_model, optimizer = accelerator.prepare(cbet_model, optimizer)

    # --- Snapshot resume logic (preemptable support) ---
    # If a snapshot exists (job was preempted and requeued), restore model,
    # optimizer, epoch counter, and bookkeeping so training continues seamlessly.
    start_epoch = 0
    resume_metrics_history = None
    resume_reward_history = None
    resume_best_eval_metric = None
    if snapshot_path.exists():
        logger.info(f"Resuming from snapshot: {snapshot_path}")
        snapshot = load_snapshot(snapshot_path, cbet_model, optimizer, accelerator)
        start_epoch = snapshot["epoch"] + 1
        resume_metrics_history = snapshot["metrics_history"]
        resume_reward_history = snapshot["reward_history"]
        resume_best_eval_metric = snapshot.get("best_eval_metric", float("-inf"))
        logger.info(f"Resumed from epoch {snapshot['epoch']}, will start at epoch {start_epoch}")

    # No need to create directory or wait - Hydra already set up the working directory
    accelerator.wait_for_everyone()


    ##############################################
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

    ##############################################

    # dataset/video already built above (before get_all_actions()); reuse them here
    use_libero_goal = cfg.data.get("use_libero_goal", False)

    precompute_embeddings = cfg.get("precompute_embeddings", True)
    # the cached embeddings live in host RAM, not VRAM, and dominate it: fp16
    # halves that at no cost to a frozen encoder's conditioning features
    embedding_fp16 = cfg.get("embedding_fp16", False)
    embedding_dtype = torch.float16 if embedding_fp16 else None
    # "memmap" keeps the cache in a file instead, so host RAM stops being the
    # ceiling on dataset size and the encoder pass is reused across runs
    use_memmap = precompute_embeddings and cfg.get("embedding_cache", "ram") == "memmap"
    precompute_start = time.perf_counter()
    if use_memmap:
        cache_dir = cfg.get("embedding_cache_dir", None) or (
            Path(cfg.env_vars.dataset_root) / "embedding_cache"
        )
        np_dtype = np.float16 if embedding_fp16 else np.float32
        cache_key = embedding_cache_key(
            OmegaConf.to_yaml(cfg.encoder, resolve=True),
            cfg.dataset.data_directory,
            cfg.dataset.get("subset_fraction", None),
            np_dtype().dtype.name,
            use_libero_goal,
        )
        embedded = FileEmbeddingDataset(
            encoder, dataset, cache_dir, cache_key,
            dtype=np_dtype, embed_goal=use_libero_goal,
        )
        # split after embedding: random_split_traj only sees len(dataset), which
        # is unchanged, so a given seed picks the same episodes either way
        train_data, test_data = split_traj_datasets(
            embedded,
            train_fraction=cfg.train_fraction,
            random_seed=cfg.seed,
        )
    else:
        train_data, test_data = split_traj_datasets(
            dataset,
            train_fraction=cfg.train_fraction,
            random_seed=cfg.seed,
        )
        if precompute_embeddings:
            train_data = TrajectoryEmbeddingDataset(
                encoder, train_data, device='cpu', embed_goal=use_libero_goal,
                dtype=embedding_dtype,
            )
            test_data = TrajectoryEmbeddingDataset(
                encoder, test_data, device='cpu', embed_goal=use_libero_goal,
                dtype=embedding_dtype,
            )
    precompute_s = time.perf_counter() - precompute_start
    print(f"Embedding precompute took {precompute_s:.1f} s")
    traj_slicer_kwargs = {
        "window": cfg.data.window_size,
        "action_window": cfg.data.action_window_size,
        "vqbet_get_future_action_chunk": cfg.data.vqbet_get_future_action_chunk,
        "future_conditional": (cfg.data.goal_conditional == "future"),
        "min_future_sep": cfg.data.action_window_size,
        "future_seq_len": cfg.data.future_seq_len,
        "use_libero_goal": use_libero_goal,
    }

    train_data = TrajectorySlicerDataset(train_data, **traj_slicer_kwargs)
    test_data = TrajectorySlicerDataset(test_data, **traj_slicer_kwargs)
    num_workers = cfg.get("num_workers", 0)
    loader_kwargs = dict(
        batch_size=cfg.gpu_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.get("prefetch_factor", 4)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **loader_kwargs)
    logger.info(f"dataloader batch size: {cfg.gpu_batch_size}")

    train_loader = accelerator.prepare(train_loader)
    test_loader = accelerator.prepare(test_loader)

    env_fn = lambda: hydra.utils.instantiate(cfg.env.gym)
    env = SubprocVectorEnv([env_fn for _ in range(cfg.num_envs)])
    if "use_libero_goal" in cfg.data:
        with torch.no_grad():
            # calculate goal embeddings for each task
            goals_cache = []
            for i in range(10):
            # for i in range(len(dataset) // 50):
                idx = i * 50
                last_obs, _, _ = dataset.get_frames(idx, [-1])  # 1 V C H W
                last_obs = last_obs.to(cfg.device)
                embd = encoder(last_obs)[0]  # V P E
                assert embd.ndim == 3, "expect V P E here"
                embd = einops.rearrange(embd, "V P E -> (V P) E") # don't flatten patch dim
                goals_cache.append(embd)

        def goal_fn(goal_idx):
            return goals_cache[goal_idx]
    else:
        empty_tensor = torch.zeros(1)

        def goal_fn(goal_idx):
            return empty_tensor

    # train_data/test_data hold the embeddings now, and the LIBERO-goal branch
    # above (if any) already read the raw frames it needs from dataset, so the
    # prefetched raw episodes can be freed instead of staying resident for the run.
    if precompute_embeddings and getattr(dataset, "prefetch", False):
        dataset.obses = None

    @torch.no_grad()
    def eval_on_env(
        cfg,
        num_evals=cfg.num_env_evals,
        videorecorder=None,
        epoch=None,
    ):
        def embed(enc, obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device)  # N V C H W
            return einops.rearrange(enc(obs), "N V P E -> N (V P) E")
        assert num_evals % cfg.num_envs == 0, "num_evals must be multiple of num_envs"

        avg_reward = 0
        action_list = []
        completion_id_list = []
        avg_max_coverage = []
        avg_final_coverage = []
        env.seed([cfg.seed + i for i in range(cfg.num_envs)])
        num_batches = num_evals // cfg.num_envs
        for goal_idx in range(num_batches):
            action_list = []
            if videorecorder is not None:
                videorecorder.init(enabled=True)
            obs_stack = deque(maxlen=cfg.eval_window_size)
            if cfg.env.gym.id == "pusht":
                # PushTEnv.reset rebuilds its RNG from the seed, so each batch needs its own seeds
                base_seed = cfg.seed + goal_idx * cfg.num_envs
                env.seed([base_seed + j for j in range(cfg.num_envs)])
            this_obs = env.reset(goal_idx=goal_idx)  # N V C H W
            print(f"Eval on goal {goal_idx}/{num_batches}, {cfg.num_envs} episodes")
            assert (
                this_obs.min() >= 0 and this_obs.max() <= 1
            ), "expect 0-1 range observation"
            this_obs_enc = embed(encoder, this_obs)  # N (V P) E. from now on V is folded into P
            obs_stack.append(this_obs_enc)
            done, total_reward = np.array([False]), 0
            goal = goal_fn(goal_idx)  # V C H W
            while not done.all():
                obs = torch.stack(tuple(obs_stack)).float().to(cfg.device)
                obs = einops.rearrange(obs, "T N P E -> N T P E")
                goal = torch.as_tensor(goal, dtype=torch.float32, device=cfg.device)
                # goal = embed(encoder, goal)
                goal = einops.repeat(goal, '... -> N T ...', N=cfg.num_envs, T=cfg.eval_window_size)
                action, _, _ = cbet_model(obs, goal, None)
                # action was T, Chunk, Action_dim for single env
                # now it's N, T, C, A for vector env
                if use_diffusion:
                    for t in range(action.shape[1]):
                        exec_action = action[:, t].cpu().detach().numpy()
                        this_obs, reward, done, info = env.step(exec_action)
                        obs_stack.append(embed(encoder, this_obs))
                        if videorecorder.enabled:
                            videorecorder.record(info[0]["image"])
                        total_reward += reward.sum()
                        goal = goal_fn(goal_idx)
                        if done.all():
                            break
                else:
                    if cfg.action_window_size > 1:
                        action_list.append(action[:, -1].cpu().detach().numpy())
                        if len(action_list) > cfg.action_window_size:
                            action_list = action_list[1:]
                        curr_action = np.array(action_list)  # W, N, C, A
                        curr_action = curr_action.mean(axis=0)[:, 0]  # N, A
                        new_action_list = []
                        for a_chunk in action_list:
                            new_action_list.append(
                                np.concatenate(
                                    (a_chunk[:, 1:], np.zeros((a_chunk.shape[0], 1, a_chunk.shape[-1]))), axis=1
                                )
                            )
                            # ! better but not tested yet:
                            # new_action_list.append(np.roll(a_chunk, -1, axis=1))
                        action_list = new_action_list
                    else:
                        curr_action = action[:, -1, 0, :].cpu().detach().numpy()

                    this_obs, reward, done, info = env.step(curr_action)
                    this_obs_enc = embed(encoder, this_obs)
                    obs_stack.append(this_obs_enc)

                    if videorecorder.enabled:
                        videorecorder.record(info[0]["image"])
                    total_reward += reward.sum()
                goal = goal_fn(goal_idx)
            avg_reward += total_reward
            if cfg.env.gym.id == "pusht":
                avg_max_coverage += [info[i]["max_coverage"] for i in range(len(info))]
                avg_final_coverage += [info[i]["final_coverage"] for i in range(len(info))]
            elif cfg.env.gym.id in ["blockpush", "cube"]:
                avg_max_coverage += [info[i]["moved"] for i in range(len(info))]
                avg_final_coverage += [info[i]["entered"] for i in range(len(info))]
            completion_id_list += [info[i]["all_completions_ids"] for i in range(len(info))]
            videorecorder.save("eval_{}_{}.mp4".format(epoch, goal_idx))
        return (
            avg_reward / num_evals,
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    # Restore bookkeeping from snapshot when resuming, else start fresh.
    metrics_history = resume_metrics_history if resume_metrics_history is not None else []
    reward_history = resume_reward_history if resume_reward_history is not None else []

    def save_model(tag, force=False):
        # force=True always saves (init/final); otherwise gated by cfg.save_ckpt.
        if not accelerator.is_main_process:
            return
        if not (force or cfg.save_ckpt):
            return
        print("cbet_model type", type(cbet_model))
        if hasattr(cbet_model, "module"):
            cbet_model_raw = accelerator.unwrap_model(cbet_model)
            torch.save(cbet_model_raw, "{}/model_{}.pt".format(save_path, tag))
        else:
            torch.save(cbet_model, "{}/model_{}.pt".format(save_path, tag))

    # Only save the init checkpoint on a fresh run, not when resuming.
    if start_epoch == 0:
        save_model("init", force=True)

    # Track the best env-specific eval metric seen so far; only save a ckpt
    # when an eval beats all previous ones (see final_eval_on_env below).
    # Restore it from the snapshot when resuming so the "best so far" gate
    # stays consistent across preemption.
    best_eval_metric = (
        resume_best_eval_metric if resume_best_eval_metric is not None else float("-inf")
    )

    # x-axis of every logged series: optimizer steps done so far. One step per
    # batch, so a resumed run at start_epoch continues on the same step values.
    steps_per_epoch = len(train_loader)
    metrics_logger = MetricsLogger(
        save_path / "tensorboard",
        accelerator.is_main_process,
        purge_step=start_epoch * steps_per_epoch if start_epoch > 0 else None,
    )
    if start_epoch == 0:
        metrics_logger.log_config(cfg)
        metrics_logger.log_scalars({"time/precompute_s": precompute_s}, step=0)
    # peak of the setup phase (embedding precompute), resets the counters
    metrics_logger.log_peak_vram(step=start_epoch * steps_per_epoch)

    def set_inference_steps(num_inference_steps):
        # denoising steps for rollouts; set on the model object in use, so it
        # also applies to checkpoints loaded through load_path
        if use_diffusion:
            accelerator.unwrap_model(cbet_model).set_inference_steps(num_inference_steps)

    def rollout_scalars(metrics):
        return {"rollout/{}".format(k.replace(" ", "_")): v for k, v in metrics.items()}

    # per-step data/GPU timers sync the GPU twice per step, which slows training
    log_step_times = cfg.get("log_step_times", False)

    for epoch in tqdm.trange(start_epoch, cfg.epochs):
        epoch_start_step = epoch * steps_per_epoch
        accelerator.wait_for_everyone()
        cbet_model.eval()
        if (epoch + 1) % cfg.eval_on_env_freq == 0:
            set_inference_steps(cfg.get("rollout_inference_steps", 100))
            rollout_start = time.perf_counter()
            avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
                cfg,
                videorecorder=video,
                epoch=epoch,
            )
            metrics_logger.log_scalars(
                {"time/rollout_s": time.perf_counter() - rollout_start},
                step=epoch_start_step, epoch=epoch,
            )
            reward_history.append(avg_reward)
            with open("{}/completion_idx_{}.json".format(save_path, epoch), "wb") as fp:
                pickle.dump(completion_id_list, fp)
            metrics_logger.log_scalars(
                {"rollout/avg_reward": avg_reward}, step=epoch_start_step, epoch=epoch
            )
            metrics = None
            if cfg.env.gym.id in ["pusht", "blockpush", "cube"]:
                metric_final = (
                    "final coverage" if cfg.env.gym.id == "pusht" else "entered"
                )
                metric_max = "max coverage" if cfg.env.gym.id == "pusht" else "moved"
                metrics = {
                    f"{metric_final} mean": sum(final_coverage) / len(final_coverage),
                    f"{metric_final} max": max(final_coverage),
                    f"{metric_final} min": min(final_coverage),
                    f"{metric_max} mean": sum(max_coverage) / len(max_coverage),
                    f"{metric_max} max": max(max_coverage),
                    f"{metric_max} min": min(max_coverage),
                }
                print("final coverage mean: ", sum(final_coverage) / len(final_coverage))
                metrics_logger.log_scalars(
                    rollout_scalars(metrics), step=epoch_start_step, epoch=epoch
                )
                metrics_history.append(metrics)

            # Save the ckpt only when this eval beats all previous evals on the
            # env-specific metric used to judge performance.
            if cfg.env.gym.id == "pusht":
                current_eval_metric = metrics["final coverage mean"]
            elif cfg.env.gym.id in ["blockpush", "cube"]:
                current_eval_metric = metrics["entered mean"]
            else:  # libero_goal
                current_eval_metric = avg_reward
            if current_eval_metric > best_eval_metric:
                print(
                    f"Eval metric {current_eval_metric} > prev best {best_eval_metric}, "
                    f"saving ckpt to {save_path}/model_{epoch}.pt"
                )
                best_eval_metric = current_eval_metric
                save_model(epoch)
            else:
                print(
                    f"Eval metric {current_eval_metric} <= prev best {best_eval_metric}, "
                    f"NOT saving ckpt"
                )

            # Synchronize all processes after eval_on_env with extended timeout
            logger.info(f"Process {accelerator.local_process_index} finished eval_on_env, waiting for others...")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            else:
                accelerator.wait_for_everyone()
            logger.info(f"Process {accelerator.local_process_index} synchronized after eval_on_env")

        if epoch % cfg.eval_freq == 0:
            eval_start = time.perf_counter()
            total_loss = 0
            action_diff = 0
            action_diff_tot = 0
            action_diff_mean_res1 = 0
            action_diff_mean_res2 = 0
            action_diff_max = 0
            eval_sums = defaultdict(float)
            with torch.no_grad():
                for data in test_loader:
                    obs, act, goal = (x.to(cfg.device, non_blocking=True) for x in data)
                    if not precompute_embeddings:
                        obs = encoder(obs)  # N T V P E
                        if use_libero_goal:
                            goal = encoder(goal)  # N T V P E
                    obs, goal = obs.float(), goal.float()
                    assert obs.ndim == 5, "expect N T V P E for obs"
                    assert goal.ndim == 5, "expect N T V P E for goals"
                    obs = einops.rearrange(obs, "N T V P E -> N T (V P) E") # keep the patch dim
                    goal = einops.rearrange(goal, "N T V P E -> N T (V P) E")
                    predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
                    total_loss += loss.item()
                    for x, y in loss_dict.items():
                        eval_sums[x] += y
                    if not use_diffusion:
                        action_diff += loss_dict["action_diff"]
                        action_diff_tot += loss_dict["action_diff_tot"]
                        action_diff_mean_res1 += loss_dict["action_diff_mean_res1"]
                        action_diff_mean_res2 += loss_dict["action_diff_mean_res2"]
                        action_diff_max += loss_dict["action_diff_max"]
            print(f"Test loss: {total_loss / len(test_loader)}")
            eval_scalars = {
                "eval/{}".format(x): y / len(test_loader) for x, y in eval_sums.items()
            }
            if not use_diffusion:
                eval_scalars.update({
                    "eval/epoch_wise_action_diff": action_diff,
                    "eval/epoch_wise_action_diff_tot": action_diff_tot,
                    "eval/epoch_wise_action_diff_mean_res1": action_diff_mean_res1,
                    "eval/epoch_wise_action_diff_mean_res2": action_diff_mean_res2,
                    "eval/epoch_wise_action_diff_max": action_diff_max,
                })
            eval_scalars["time/eval_s"] = time.perf_counter() - eval_start
            metrics_logger.log_scalars(eval_scalars, step=epoch_start_step, epoch=epoch)

        accelerator.wait_for_everyone()
        cbet_model.train()
        train_loss = 0
        train_sums = defaultdict(float)
        step_times = {}
        time_sums = defaultdict(float)
        train_start = time.perf_counter()
        if log_step_times:
            data_start = time.perf_counter()
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            obs, act, goal = (x.to(cfg.device, non_blocking=True) for x in data)
            if log_step_times:
                torch.cuda.synchronize()
                step_start = time.perf_counter()
                # data_s: loader (slice, collate, pin) plus the copy to the GPU
                step_times = {"time/data_s": step_start - data_start}
            if not precompute_embeddings:
                obs = encoder(obs)  # N T V P E
                if use_libero_goal:
                    goal = encoder(goal)  # N T V P E
            # fp16 cache: upcast on the GPU, so the copy above moves half the bytes
            obs, goal = obs.float(), goal.float()
            obs = einops.rearrange(obs, "N T V P E -> N T (V P) E")
            goal = einops.rearrange(goal, "N T V P E -> N T (V P) E")
            predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
            train_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()

            if use_diffusion:
                if hasattr(cbet_model, "module"):
                    cbet_model_raw = accelerator.unwrap_model(cbet_model)
                    cbet_model_raw.ema_step()
                else:
                    cbet_model.ema_step()
            if log_step_times:
                torch.cuda.synchronize()
                # step_s: forward, backward, optimizer and EMA on the GPU
                step_times["time/step_s"] = time.perf_counter() - step_start

            for x, y in loss_dict.items():
                train_sums[x] += y
            for x, y in step_times.items():
                time_sums[x] += y
            metrics_logger.log_scalars(
                {**{"train/{}".format(x): y for x, y in loss_dict.items()}, **step_times},
                step=epoch_start_step + i + 1,
                epoch=epoch,
            )
            if log_step_times:
                data_start = time.perf_counter()

        if hasattr(cbet_model, "module"):
            if hasattr(accelerator.unwrap_model(cbet_model), "finish_epoch"):
                accelerator.unwrap_model(cbet_model).finish_epoch()
        else:
            if hasattr(cbet_model, "finish_epoch"):
                cbet_model.finish_epoch()
        print(f"Train loss: {train_loss / len(train_loader)}")
        epoch_end_step = (epoch + 1) * steps_per_epoch
        metrics_logger.log_scalars(
            {"train/epoch_{}".format(x): y / len(train_loader) for x, y in train_sums.items()},
            step=epoch_end_step,
            epoch=epoch,
        )
        epoch_times = {"time/train_epoch_s": time.perf_counter() - train_start}
        if log_step_times:
            epoch_times["time/epoch_data_s"] = time_sums["time/data_s"]
            epoch_times["time/epoch_step_s"] = time_sums["time/step_s"]
        metrics_logger.log_scalars(epoch_times, step=epoch_end_step, epoch=epoch)
        # covers this epoch's rollouts, test-set eval and training steps
        metrics_logger.log_peak_vram(step=epoch_end_step, epoch=epoch)

        # Save a resume snapshot every epoch (preemptable support). This is
        # independent of the best-eval-gated model_{epoch}.pt above: the
        # snapshot always captures the latest training state so a preempted
        # job can be requeued and continue from the next epoch.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            snapshot_start = time.perf_counter()
            save_snapshot(save_path, cbet_model, optimizer, accelerator, epoch,
                          wandb_run_id, metrics_history, reward_history,
                          best_eval_metric)
            metrics_logger.log_scalars(
                {"time/snapshot_s": time.perf_counter() - snapshot_start},
                step=epoch_end_step, epoch=epoch,
            )

    set_inference_steps(cfg.get("final_inference_steps", 100))
    final_eval_start = time.perf_counter()
    avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
        cfg,
        num_evals=cfg.num_final_evals,
        videorecorder=video,
        epoch=cfg.epochs,
    )
    metrics_logger.log_scalars(
        {"time/final_eval_s": time.perf_counter() - final_eval_start},
        step=cfg.epochs * steps_per_epoch, epoch=cfg.epochs,
    )

    # Synchronize all processes after final eval_on_env
    logger.info(f"Process {accelerator.local_process_index} finished final eval_on_env, waiting for others...")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        accelerator.wait_for_everyone()
    logger.info(f"Process {accelerator.local_process_index} synchronized after final eval_on_env")

    print(f"Final eval, always saving ckpt to {save_path}/model_final.pt")
    save_model("final", force=True)
    reward_history.append(avg_reward)
    metrics_logger.log_scalars(
        {"rollout/avg_reward": avg_reward}, step=cfg.epochs * steps_per_epoch, epoch=cfg.epochs
    )
    if cfg.env.gym.id in ["pusht", "blockpush", "cube"]:
        metric_final = "final coverage" if cfg.env.gym.id == "pusht" else "entered"
        metric_max = "max coverage" if cfg.env.gym.id == "pusht" else "moved"
        metrics = {
            f"{metric_final} mean": sum(final_coverage) / len(final_coverage),
            f"{metric_final} max": max(final_coverage),
            f"{metric_final} min": min(final_coverage),
            f"{metric_max} mean": sum(max_coverage) / len(max_coverage),
            f"{metric_max} max": max(max_coverage),
            f"{metric_max} min": min(max_coverage),
        }
        metrics_logger.log_scalars(
            rollout_scalars(metrics), step=cfg.epochs * steps_per_epoch, epoch=cfg.epochs
        )
        metrics_history.append(metrics)

    with open("{}/completion_idx_final.json".format(save_path), "wb") as fp:
        pickle.dump(completion_id_list, fp)
    if cfg.env.gym.id == "pusht":
        final_eval_on_env = max([x["final coverage mean"] for x in metrics_history])
    elif cfg.env.gym.id == "blockpush" or cfg.env.gym.id == "cube":
        final_eval_on_env = max([x["entered mean"] for x in metrics_history])
    else:  # libero_goal and anything else without a coverage-style metric
        final_eval_on_env = max(reward_history)
    metrics_logger.log_scalars(
        {"rollout/best_final_eval": final_eval_on_env},
        step=cfg.epochs * steps_per_epoch,
        epoch=cfg.epochs,
    )
    metrics_logger.close()
    return final_eval_on_env


if __name__ == "__main__":
    main()
