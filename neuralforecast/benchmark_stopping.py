"""Validation-only checkpoint selection for commodity Phase 2."""

from pathlib import Path
import math
import torch


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

    def summary(self):
        return {
            "best_step": self.best_step,
            "actual_steps": self.last_step,
            "best_validation_loss": self.best,
            "stop_reason": "early_stopping" if self.stopped else "max_steps",
        }
