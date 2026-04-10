"""CLI for trainpulse."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


def _build_cli():  # type: ignore[no-untyped-def]
    """Build the CLI. Deferred import so click/rich are optional."""
    try:
        import click
    except ImportError:
        raise SystemExit(
            "CLI dependencies required: pip install trainpulse[cli]"
        )

    @click.group()
    @click.version_option(package_name="trainpulse")
    def cli() -> None:
        """trainpulse — lightweight training health monitor."""

    @cli.command()
    @click.argument("log_file", type=click.Path(exists=True))
    @click.option("--json-out", "-o", type=click.Path(), default=None, help="Save report as JSON.")
    @click.option("--loss-key", default="loss", help="Key for loss in log entries.")
    @click.option("--grad-key", default="grad_norm", help="Key for gradient norm.")
    @click.option("--lr-key", default="learning_rate", help="Key for learning rate.")
    @click.option("--step-key", default="step", help="Key for step number.")
    def analyze(
        log_file: str,
        json_out: Optional[str],
        loss_key: str,
        grad_key: str,
        lr_key: str,
        step_key: str,
    ) -> None:
        """Analyze a training log file (JSONL format).

        Each line should be a JSON object with at least a step and loss field.
        """
        from trainpulse.monitor import Monitor
        from trainpulse.report import format_report_rich, format_report_text, save_json

        monitor = Monitor()
        path = Path(log_file)

        n_lines = 0
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = entry.get(step_key, n_lines)

            if loss_key in entry:
                monitor.log("loss", step, float(entry[loss_key]))
            if grad_key in entry:
                monitor.log("grad_norm", step, float(entry[grad_key]))
            if lr_key in entry:
                monitor.log("learning_rate", step, float(entry[lr_key]))

            n_lines += 1

        if n_lines == 0:
            click.echo("No log entries found.", err=True)
            sys.exit(1)

        report = monitor.report()

        try:
            output = format_report_rich(report)
        except Exception:
            output = format_report_text(report)
        click.echo(output)

        if json_out:
            save_json(report, json_out)
            click.echo(f"Report saved to {json_out}")

    @cli.command()
    @click.argument("report_file", type=click.Path(exists=True))
    def show(report_file: str) -> None:
        """Display a previously saved JSON report."""
        from trainpulse._types import Alert, AlertSeverity, TrainingReport
        from trainpulse.report import format_report_rich, format_report_text, load_json

        data = load_json(report_file)
        alerts = [
            Alert(
                step=a["step"],
                severity=AlertSeverity(a["severity"]),
                detector=a["detector"],
                message=a["message"],
                metric_name=a.get("metric_name", ""),
                metric_value=a.get("metric_value", 0.0),
            )
            for a in data.get("alerts", [])
        ]
        report = TrainingReport(
            total_steps=data["total_steps"],
            alerts=alerts,
            metrics_summary=data.get("metrics_summary", {}),
            health_score=data.get("health_score", 1.0),
        )

        try:
            output = format_report_rich(report)
        except Exception:
            output = format_report_text(report)
        click.echo(output)

    return cli


cli = _build_cli()

if __name__ == "__main__":
    cli()
