"""Validation checkpoint selection and portable training continuation state."""

from pathlib import Path
import math
import random
import numpy as np
import torch


def rng_state():
    """Capture RNGs used by benchmark training, including accelerator dropout."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    """Restore a locally produced training RNG snapshot."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


class ValidationStopper:
    """Save the best weights and stop after consecutive non-improvements.

    Args:
        directory: Directory for the best checkpoint.
        interval: Number of optimizer steps between validation checks.
        patience: Number of consecutive checks without improvement.
        metrics_callback: Optional benchmark-scoped metrics sink.
        adapters_only: Store trainable parameters only for LoRA models.
    """

    def __init__(
        self,
        directory,
        interval=10,
        patience=5,
        metrics_callback=None,
        adapters_only=False,
    ):
        if interval < 1 or patience < 1:
            raise ValueError("Validation interval and patience must be positive")
        self.path = Path(directory) / "best-validation.pt"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval, self.patience = interval, patience
        self.callback, self.adapters_only = metrics_callback, adapters_only
        self.best, self.best_step, self.bad_checks = math.inf, None, 0
        self.last_step, self.stopped = 0, False

    def update(self, model, step, loss):
        if step <= self.last_step:
            return self.stopped
        self.last_step = step
        if not math.isfinite(loss):
            raise ValueError("Non-finite validation loss")
        if loss < self.best:
            self.best, self.best_step, self.bad_checks = loss, step, 0
            names = {n for n, p in model.named_parameters() if p.requires_grad}
            weights = {
                n: v.detach().cpu().clone()
                for n, v in model.state_dict().items()
                if not self.adapters_only or n in names
            }
            torch.save(weights, self.path)
        else:
            self.bad_checks += 1
        self.stopped = self.bad_checks >= self.patience
        if self.callback:
            self.callback(
                {
                    "train/global_step": step,
                    "validation/loss": loss,
                    "validation/best_step": self.best_step,
                }
            )
        return self.stopped

    def restore(self, model):
        if self.best_step is None:
            raise RuntimeError("No validated checkpoint was produced")
        model.load_state_dict(
            torch.load(self.path, map_location="cpu", weights_only=True),
            strict=not self.adapters_only,
        )

    def state_dict(self):
        """Bundle the best weights so continuation can use a different directory."""
        return {
            "best": self.best,
            "best_step": self.best_step,
            "bad_checks": self.bad_checks,
            "last_step": self.last_step,
            "stopped": self.stopped,
            "weights": (
                torch.load(self.path, map_location="cpu", weights_only=True)
                if self.best_step is not None
                else None
            ),
        }

    def load_state_dict(self, state):
        """Restore patience and the best checkpoint without resetting either."""
        for key in ("best", "best_step", "bad_checks", "last_step", "stopped"):
            setattr(self, key, state[key])
        if state["weights"] is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state["weights"], self.path)

    def summary(self):
        return {
            "best_step": self.best_step,
            "actual_steps": self.last_step,
            "best_validation_loss": self.best,
            "stop_reason": "early_stopping" if self.stopped else "max_steps",
        }
