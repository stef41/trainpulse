"""Report formatting and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trainpulse._types import AlertSeverity, TrainingReport


def report_to_dict(report: TrainingReport) -> dict[str, Any]:
    """Convert a report to a JSON-serializable dict."""
    return {
        "total_steps": report.total_steps,
        "health_score": round(report.health_score, 4),
        "is_healthy": report.is_healthy,
        "n_warnings": report.n_warnings,
        "n_critical": report.n_critical,
        "alerts": [
            {
                "step": a.step,
                "severity": a.severity.value,
                "detector": a.detector,
                "message": a.message,
                "metric_name": a.metric_name,
                "metric_value": a.metric_value,
            }
            for a in report.alerts
        ],
        "metrics_summary": report.metrics_summary,
    }


def save_json(report: TrainingReport, path: str | Path) -> None:
    """Save report as JSON."""
    Path(path).write_text(json.dumps(report_to_dict(report), indent=2))


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a report dict from JSON."""
    return json.loads(Path(path).read_text())  # type: ignore[no-any-return]


def format_report_text(report: TrainingReport) -> str:
    """Format report as plain text."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("TRAINING HEALTH REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Total steps:   {report.total_steps}")
    lines.append(f"  Health score:  {report.health_score:.1%}")
    lines.append(f"  Status:        {'HEALTHY' if report.is_healthy else 'UNHEALTHY'}")
    lines.append(f"  Warnings:      {report.n_warnings}")
    lines.append(f"  Critical:      {report.n_critical}")
    lines.append("")

    if report.metrics_summary:
        lines.append("-" * 60)
        lines.append("METRICS SUMMARY")
        lines.append("-" * 60)
        for name, stats in sorted(report.metrics_summary.items()):
            lines.append(f"  {name}:")
            lines.append(
                f"    min={stats['min']:.6f}  max={stats['max']:.6f}  "
                f"mean={stats['mean']:.6f}  last={stats['last']:.6f}"
            )
        lines.append("")

    if report.alerts:
        lines.append("-" * 60)
        lines.append("ALERTS")
        lines.append("-" * 60)
        for a in report.alerts:
            prefix = "!!" if a.severity == AlertSeverity.CRITICAL else " >"
            lines.append(f"  {prefix} Step {a.step}: [{a.severity.value.upper()}] {a.message}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_report_rich(report: TrainingReport) -> str:
    """Format report using rich (returns string from Console capture)."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        return format_report_text(report)

    console = Console(record=True, width=90)

    # Header
    status_color = "green" if report.is_healthy else "red"
    status_text = "HEALTHY" if report.is_healthy else "UNHEALTHY"
    header = Text()
    header.append(f"Health Score: {report.health_score:.1%}", style="bold")
    header.append("  |  ", style="dim")
    header.append(status_text, style=f"bold {status_color}")
    header.append("  |  ", style="dim")
    header.append(f"{report.total_steps} steps", style="cyan")
    header.append("  |  ", style="dim")
    header.append(f"{report.n_warnings} warnings", style="yellow")
    header.append("  |  ", style="dim")
    header.append(f"{report.n_critical} critical", style="red")

    console.print(Panel(header, title="[bold]trainpulse[/bold]", border_style="blue"))

    # Metrics table
    if report.metrics_summary:
        table = Table(title="Metrics Summary", show_lines=False)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Last", justify="right")
        table.add_column("Count", justify="right", style="dim")

        for name, stats in sorted(report.metrics_summary.items()):
            table.add_row(
                name,
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
                f"{stats['mean']:.6f}",
                f"{stats['last']:.6f}",
                str(int(stats["count"])),
            )
        console.print(table)
        console.print()

    # Alerts table
    if report.alerts:
        alert_table = Table(title="Alerts", show_lines=False)
        alert_table.add_column("Step", justify="right", style="dim")
        alert_table.add_column("Severity", no_wrap=True)
        alert_table.add_column("Detector", style="cyan")
        alert_table.add_column("Message")

        for a in report.alerts:
            sev_style = {
                AlertSeverity.INFO: "blue",
                AlertSeverity.WARNING: "yellow",
                AlertSeverity.CRITICAL: "red bold",
            }[a.severity]
            alert_table.add_row(
                str(a.step),
                Text(a.severity.value.upper(), style=sev_style),
                a.detector,
                a.message,
            )
        console.print(alert_table)

    return console.export_text()
