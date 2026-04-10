"""Basic training monitoring with trainpulse.

Demonstrates how to use the Monitor class to track loss and gradient
norms during a simulated training loop, then generate a report.
"""

import math
import random

from trainpulse import Monitor, MonitorConfig


def simulated_training_step(step: int) -> tuple[float, float]:
    """Simulate a training step returning (loss, grad_norm)."""
    # Loss that generally decreases with noise
    base_loss = 2.0 * math.exp(-step / 300) + 0.3
    noise = random.gauss(0, 0.05)
    loss = max(0.01, base_loss + noise)

    # Gradient norm that settles over time
    grad_norm = random.uniform(0.5, 3.0) * math.exp(-step / 500) + 0.1
    return loss, grad_norm


def main() -> None:
    config = MonitorConfig(
        loss_spike_threshold=4.0,
        grad_norm_threshold=50.0,
        plateau_patience=80,
    )
    monitor = Monitor(config=config)

    num_steps = 500
    print(f"Running {num_steps} simulated training steps...")

    for step in range(num_steps):
        loss, grad_norm = simulated_training_step(step)

        loss_alerts = monitor.log("loss", step, loss)
        grad_alerts = monitor.log("grad_norm", step, grad_norm)

        for alert in loss_alerts + grad_alerts:
            print(f"  ALERT at step {step}: {alert}")

    report = monitor.report()
    print(f"\n{'='*50}")
    print(f"Training Report")
    print(f"{'='*50}")
    print(f"Total steps: {report.total_steps}")
    print(f"Total alerts: {len(report.alerts)}")
    for severity, count in report.alerts_by_severity.items():
        print(f"  {severity}: {count}")
    print(f"Final loss: {report.final_values.get('loss', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
