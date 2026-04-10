"""Framework integration callbacks."""

from __future__ import annotations

from typing import Any

from trainpulse._types import MonitorConfig
from trainpulse.monitor import Monitor


class TrainingCallback:
    """Generic callback that wraps a Monitor for any training loop.

    Usage::

        cb = TrainingCallback()
        for step in range(num_steps):
            cb.on_step_begin(step)
            loss = train_step()
            grad_norm = get_grad_norm()
            cb.on_step_end(step, loss=loss, grad_norm=grad_norm, lr=optimizer.lr)
        report = cb.report()
    """

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.monitor = Monitor(config)

    def on_step_begin(self, step: int) -> None:
        self.monitor.step_start()

    def on_step_end(
        self,
        step: int,
        loss: float | None = None,
        grad_norm: float | None = None,
        lr: float | None = None,
        **extra_metrics: float,
    ) -> None:
        self.monitor.step_end(step)
        if loss is not None:
            self.monitor.log("loss", step, loss)
        if grad_norm is not None:
            self.monitor.log("grad_norm", step, grad_norm)
        if lr is not None:
            self.monitor.log("learning_rate", step, lr)
        for name, value in extra_metrics.items():
            self.monitor.log(name, step, value)

    def report(self) -> Any:
        return self.monitor.report()


def make_pytorch_hooks(
    model: Any,
    monitor: Monitor,
) -> list:
    """Register backward hooks on a PyTorch model to track gradient norms.

    Returns a list of hook handles for cleanup.

    Usage::

        monitor = Monitor()
        hooks = make_pytorch_hooks(model, monitor)
        # ... training loop ...
        for h in hooks:
            h.remove()
    """
    try:
        import torch  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        raise ImportError("PyTorch is required: pip install trainpulse[torch]")

    handles = []
    _step_counter = {"step": 0}

    def _grad_hook(module: Any, grad_input: Any, grad_output: Any) -> None:
        total_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        monitor.log("grad_norm", _step_counter["step"], total_norm)

    for module in model.modules():
        # Only hook leaf modules to avoid double counting
        if len(list(module.children())) == 0:
            h = module.register_full_backward_hook(_grad_hook)
            handles.append(h)

    class _HookManager:
        """Manages hooks and step counter."""

        def __init__(self, handles: list, counter: dict) -> None:
            self._handles = handles
            self._counter = counter

        def set_step(self, step: int) -> None:
            self._counter["step"] = step

        def remove(self) -> None:
            for h in self._handles:
                h.remove()

    manager = _HookManager(handles, _step_counter)
    return [manager]


def make_hf_callback(
    config: MonitorConfig | None = None,
) -> Any:
    """Create a HuggingFace Trainer callback.

    Usage::

        from transformers import Trainer
        from trainpulse.callbacks import make_hf_callback

        cb = make_hf_callback()
        trainer = Trainer(..., callbacks=[cb])
        trainer.train()
        report = cb.trainpulse_monitor.report()

    Returns a TrainerCallback subclass instance.
    """
    try:
        from transformers import (  # type: ignore[import-untyped]
            TrainerCallback,
            TrainerControl,
            TrainerState,
            TrainingArguments,
        )
    except ImportError:
        raise ImportError(
            "HuggingFace transformers is required: pip install transformers"
        )

    monitor = Monitor(config)

    class _TrainpulseCallback(TrainerCallback):
        def __init__(self) -> None:
            self.trainpulse_monitor = monitor

        def on_step_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ) -> None:
            monitor.step_start()

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kwargs: Any,
        ) -> None:
            if logs is None:
                return
            step = state.global_step
            if "loss" in logs:
                monitor.log("loss", step, logs["loss"])
            if "learning_rate" in logs:
                monitor.log("learning_rate", step, logs["learning_rate"])
            if "grad_norm" in logs:
                monitor.log("grad_norm", step, logs["grad_norm"])

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ) -> None:
            monitor.step_end(state.global_step)

    return _TrainpulseCallback()
