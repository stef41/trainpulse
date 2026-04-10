"""Weights & Biases callback integration for trainpulse."""

from __future__ import annotations

from typing import Any, Dict, Optional

from trainpulse._types import Alert


class WandbCallback:
    """Callback that logs training metrics and alerts to Weights & Biases.

    W&B is imported lazily on the first call that requires it.
    If ``enabled=False``, all methods are silent no-ops.

    Usage::

        cb = WandbCallback(project="my-run")
        for step in range(num_steps):
            loss = train_step()
            cb.on_step(step, {"loss": loss})
        cb.finish()
    """

    def __init__(
        self,
        project: str = "trainpulse",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        self._project = project
        self._run_name = run_name
        self._config = config
        self._enabled = enabled
        self._wandb: Any = None
        self._run: Any = None

    def _ensure_wandb(self) -> Any:
        """Lazily import wandb and initialise a run."""
        if self._wandb is not None:
            return self._wandb
        try:
            import wandb  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "wandb is required for WandbCallback. "
                "Install it with: pip install wandb"
            )
        self._wandb = wandb
        self._run = wandb.init(
            project=self._project,
            name=self._run_name,
            config=self._config,
        )
        return self._wandb

    def on_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Log per-step metrics to W&B."""
        if not self._enabled:
            return
        wandb = self._ensure_wandb()
        wandb.log(metrics, step=step)

    def on_alert(self, alert: Alert) -> None:
        """Log a trainpulse Alert to W&B as a wandb.alert."""
        if not self._enabled:
            return
        wandb = self._ensure_wandb()
        wandb.alert(
            title=f"[{alert.severity.value.upper()}] {alert.detector}",
            text=alert.message,
        )

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log epoch-level summary metrics to W&B."""
        if not self._enabled:
            return
        wandb = self._ensure_wandb()
        prefixed = {f"epoch/{k}": v for k, v in metrics.items()}
        prefixed["epoch"] = epoch
        wandb.log(prefixed)

    def finish(self) -> None:
        """Finish the W&B run."""
        if not self._enabled:
            return
        if self._wandb is not None:
            self._wandb.finish()
