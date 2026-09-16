from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    """Writes scalars to TensorBoard and forwards them to wandb.

    All series share one x-axis, `step`: the number of optimizer steps done.
    Only the main process writes; on other processes every call is a no-op.
    """

    def __init__(self, log_dir, is_main_process, purge_step=None):
        self.enabled = is_main_process
        self.writer = None
        if self.enabled:
            # purge_step drops events a killed run wrote past the resume point
            self.writer = SummaryWriter(log_dir=str(Path(log_dir)), purge_step=purge_step)

    def log_scalars(self, scalars, step, epoch=None):
        if not self.enabled:
            return
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, float(value), global_step=step)
        extra = {"step": step} if epoch is None else {"step": step, "epoch": epoch}
        wandb.log({**scalars, **extra})

    def log_config(self, cfg):
        if not self.enabled:
            return
        hparams = {
            "lr": cfg.optim.lr,
            "weight_decay": cfg.optim.weight_decay,
            "betas": list(cfg.optim.betas),
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "rollout_inference_steps": cfg.get("rollout_inference_steps", 100),
            "final_inference_steps": cfg.get("final_inference_steps", 100),
        }
        table = "| name | value |\n| --- | --- |\n" + "".join(
            f"| {k} | {v} |\n" for k, v in hparams.items()
        )
        self.writer.add_text("hparams", table, global_step=0)
        config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        # four-space indent renders the yaml as a code block
        self.writer.add_text("config", "    " + config_yaml.replace("\n", "\n    "), global_step=0)

    def log_peak_vram(self, step, epoch=None):
        """Log the CUDA allocator peaks since the last call, then reset them."""
        if not torch.cuda.is_available():
            return
        mib = 1024 ** 2
        self.log_scalars(
            {
                "system/vram_peak_allocated_mib": torch.cuda.max_memory_allocated() / mib,
                "system/vram_peak_reserved_mib": torch.cuda.max_memory_reserved() / mib,
            },
            step,
            epoch,
        )
        torch.cuda.reset_peak_memory_stats()

    def close(self):
        if self.writer is not None:
            self.writer.close()
