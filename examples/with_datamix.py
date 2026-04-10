#!/usr/bin/env python3
"""Integration: datamix + trainpulse — create a mix recipe, monitor each phase.

Flow: Use datamix to build a multi-phase curriculum training recipe, then
simulate training with trainpulse monitoring health metrics per phase.

Install: pip install datamix trainpulse
"""
import random

try:
    from datamix import (
        MixRecipe, MixStrategy, DatasetProfile, CurriculumPhase,
        create_recipe, linear_schedule, compute_budget, profile_dataset,
    )
except ImportError:
    raise SystemExit("pip install datamix  # required for this example")

try:
    from trainpulse import Monitor, MonitorConfig, Alert, AlertSeverity
except ImportError:
    raise SystemExit("pip install trainpulse  # required for this example")


def main() -> None:
    # ── 1. Build a training recipe with datamix ──────────────────────
    print("=" * 60)
    print("STEP 1: Create a multi-dataset mix recipe (datamix)")
    print("=" * 60)
    profiles = {
        "code":    DatasetProfile(name="code", num_examples=50_000, avg_tokens=320),
        "math":    DatasetProfile(name="math", num_examples=30_000, avg_tokens=180),
        "general": DatasetProfile(name="general", num_examples=100_000, avg_tokens=210),
    }
    recipe = create_recipe(
        datasets=profiles,
        strategy=MixStrategy.PROPORTIONAL,
        token_budget=500_000_000,
    )
    for name, weight in recipe.weights.items():
        print(f"  {name}: {weight:.1%}")

    # ── 2. Define a curriculum schedule ──────────────────────────────
    print("\nSTEP 2: Define curriculum phases")
    schedule = linear_schedule(
        phases=[
            CurriculumPhase(name="warmup", datasets=["general"], steps=500),
            CurriculumPhase(name="specialise", datasets=["code", "math"], steps=1000),
            CurriculumPhase(name="blend", datasets=["code", "math", "general"], steps=500),
        ]
    )
    for phase in schedule.phases:
        print(f"  Phase '{phase.name}': {phase.steps} steps — {phase.datasets}")

    # ── 3. Simulate training with trainpulse per phase ───────────────
    print("\n" + "=" * 60)
    print("STEP 3: Monitor training health per phase (trainpulse)")
    print("=" * 60)
    monitor = Monitor(MonitorConfig(
        loss_spike_threshold=2.0,
        plateau_patience=50,
        check_nan=True,
    ))

    global_step = 0
    for phase in schedule.phases:
        print(f"\n  ── Phase: {phase.name} ({phase.steps} steps) ──")
        base_loss = {"warmup": 3.5, "specialise": 2.0, "blend": 1.5}[phase.name]
        for local_step in range(phase.steps):
            noise = random.gauss(0, 0.05)
            decay = base_loss * (1 - 0.3 * local_step / phase.steps)
            loss = max(0.01, decay + noise)
            monitor.log("loss", global_step, loss)
            monitor.log("lr", global_step, 1e-4 * (1 - global_step / 2000))
            global_step += 1

        phase_alerts = [a for a in monitor.alerts if a.step >= global_step - phase.steps]
        print(f"    Final loss ≈ {loss:.4f}")
        print(f"    Alerts in phase: {len(phase_alerts)}")

    # ── 4. Final report ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Training health summary")
    print("=" * 60)
    report = monitor.report()
    print(f"  Total steps:  {report.total_steps}")
    print(f"  Total alerts: {len(report.alerts)}")
    for alert in report.alerts[:5]:
        print(f"    [{alert.severity.value}] step {alert.step}: {alert.message}")
    print("\nCurriculum training with live monitoring complete.")


if __name__ == "__main__":
    main()
