"""Anomaly detection examples with trainpulse.

Demonstrates how trainpulse detects NaN loss, gradient explosion,
and loss plateaus in a simulated training run.
"""

import math
import random

from trainpulse import Alert, Monitor, MonitorConfig


def on_alert(alert: Alert) -> None:
    """Callback that fires on every alert."""
    print(f"  [{alert.severity.value.upper()}] step={alert.step}: {alert.message}")


def run_nan_scenario(monitor: Monitor) -> None:
    """Simulate a training run where loss goes NaN at step 30."""
    print("--- Scenario 1: NaN loss ---")
    for step in range(50):
        loss = 1.5 - step * 0.02 if step < 30 else float("nan")
        monitor.log("loss", step, loss)


def run_gradient_explosion(monitor: Monitor) -> None:
    """Simulate gradient norms that spike exponentially."""
    print("\n--- Scenario 2: Gradient explosion ---")
    for step in range(60):
        if step < 40:
            grad = random.uniform(0.5, 2.0)
        else:
            grad = 2.0 * (1.5 ** (step - 40))  # exponential blowup
        monitor.log("grad_norm", step, grad)


def run_plateau(monitor: Monitor) -> None:
    """Simulate a loss that flatlines after initial decrease."""
    print("\n--- Scenario 3: Loss plateau ---")
    for step in range(250):
        if step < 50:
            loss = 2.0 * math.exp(-step / 20) + 0.5
        else:
            loss = 0.5 + random.gauss(0, 1e-6)  # flatline
        monitor.log("loss", step, loss)


def main() -> None:
    config = MonitorConfig(
        check_nan=True,
        grad_norm_threshold=50.0,
        plateau_patience=80,
        plateau_min_delta=1e-4,
        alert_callbacks=[on_alert],
    )

    for scenario in [run_nan_scenario, run_gradient_explosion, run_plateau]:
        monitor = Monitor(config=config)
        scenario(monitor)
        alerts = monitor.alerts
        print(f"  Total alerts: {len(alerts)}\n")


if __name__ == "__main__":
    main()
