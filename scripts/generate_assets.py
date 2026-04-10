"""Generate SVG assets for README."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def generate_report_svg() -> None:
    console = Console(record=True, width=90)

    # Header panel
    header = Text()
    header.append("Health Score: 85.0%", style="bold")
    header.append("  |  ", style="dim")
    header.append("UNHEALTHY", style="bold red")
    header.append("  |  ", style="dim")
    header.append("500 steps", style="cyan")
    header.append("  |  ", style="dim")
    header.append("2 warnings", style="yellow")
    header.append("  |  ", style="dim")
    header.append("1 critical", style="red")
    console.print(Panel(header, title="[bold]trainpulse[/bold]", border_style="blue"))

    # Metrics table
    table = Table(title="Metrics Summary", show_lines=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Last", justify="right")
    table.add_column("Count", justify="right", style="dim")

    table.add_row("grad_norm", "0.012000", "87.340000", "2.341000", "1.890000", "500")
    table.add_row("learning_rate", "0.000010", "0.001000", "0.000450", "0.000100", "500")
    table.add_row("loss", "0.142000", "8.930000", "0.823000", "0.198000", "500")
    table.add_row("step_time", "0.120000", "2.340000", "0.156000", "0.148000", "500")
    console.print(table)
    console.print()

    # Alerts table
    alert_table = Table(title="Alerts", show_lines=False)
    alert_table.add_column("Step", justify="right", style="dim")
    alert_table.add_column("Severity", no_wrap=True)
    alert_table.add_column("Detector", style="cyan")
    alert_table.add_column("Message")

    alert_table.add_row("87", Text("WARNING", style="yellow"), "loss_spike", "Loss spike: 8.930 (6.2x rolling avg 1.440)")
    alert_table.add_row("203", Text("CRITICAL", style="red bold"), "nan_detector", "loss is NaN")
    alert_table.add_row("341", Text("WARNING", style="yellow"), "step_time", "Slow step: 2.34s (4.1x avg 0.57s)")
    console.print(alert_table)

    svg = console.export_svg(title="trainpulse report")
    Path("assets/report.svg").write_text(svg)
    print(f"  report.svg: {len(svg) // 1024}KB")


def generate_alerts_svg() -> None:
    console = Console(record=True, width=90)

    console.print("[dim]Step 84:[/dim]  loss=1.42  grad_norm=2.1  lr=4.5e-04")
    console.print("[dim]Step 85:[/dim]  loss=1.38  grad_norm=2.3  lr=4.5e-04")
    console.print("[dim]Step 86:[/dim]  loss=1.44  grad_norm=2.0  lr=4.5e-04")
    console.print("[bold yellow]⚠ [WARNING] Step 87: Loss spike: 8.930 (6.2x rolling avg 1.440)[/bold yellow]")
    console.print("[dim]Step 87:[/dim]  loss=8.93  grad_norm=45.2  lr=4.5e-04")
    console.print("[dim]Step 88:[/dim]  loss=3.21  grad_norm=12.1  lr=4.5e-04")
    console.print("[dim]Step 89:[/dim]  loss=2.05  grad_norm=5.4  lr=4.5e-04")
    console.print()
    console.print("[dim]Step 201:[/dim]  loss=0.89  grad_norm=1.8  lr=2.1e-04")
    console.print("[dim]Step 202:[/dim]  loss=0.91  grad_norm=1.9  lr=2.1e-04")
    console.print("[bold red]!! [CRITICAL] Step 203: loss is NaN[/bold red]")
    console.print("[dim]Step 203:[/dim]  loss=nan  grad_norm=inf  lr=2.1e-04")
    console.print("[bold red]!! [CRITICAL] Step 203: grad_norm is Inf[/bold red]")

    svg = console.export_svg(title="trainpulse alerts")
    Path("assets/alerts.svg").write_text(svg)
    print(f"  alerts.svg: {len(svg) // 1024}KB")


if __name__ == "__main__":
    print("Generating trainpulse assets...")
    generate_report_svg()
    generate_alerts_svg()
    print("Done!")
