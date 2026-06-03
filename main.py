"""
Hypestrat — HYPE/USDC Quantitative Trading System
Entry point for live trading, backtesting, and performance reporting.

Usage:
  python main.py live          # Start live trading
  python main.py backtest      # Run walk-forward backtest
  python main.py report        # Print performance report
  python main.py schedule      # Print compound growth schedule
  python main.py --help        # Show all options

Growth Target:
  $5,000 → $1,000,000 in 4 years
  Required CAGR:    ~276%
  Required Monthly: ~11.34%
  Strategy:         7-signal ensemble + HMM regime + Kelly sizing + strict risk management
"""
import os
import sys
import time
from datetime import datetime, timezone

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import CONFIG
from utils.logger import logger, setup_logger
from analytics.performance import PerformanceAnalytics

console = Console()


def print_banner() -> None:
    console.print(Panel.fit(
        "[bold green]HYPESTRAT[/bold green] — [cyan]HYPE/USDC Quantitative Trading System[/cyan]\n"
        "[yellow]Target: $5,000 → $1,000,000 in 4 years (276% CAGR)[/yellow]\n"
        "[dim]7-Signal Ensemble | HMM Regime | GARCH Vol | LSTM+XGB ML | Kelly Sizing[/dim]",
        title="[bold]Hyperliquid Perp Trading[/bold]",
        border_style="bright_blue",
    ))


@click.group()
@click.option("--log-level", default="INFO", help="Log level (DEBUG/INFO/WARNING)")
def cli(log_level: str) -> None:
    """Hypestrat — HYPE/USDC algorithmic trading system."""
    setup_logger(level=log_level, log_file=CONFIG.log.file)
    print_banner()


@cli.command()
@click.option("--interval", default=900, help="Loop interval in seconds (default 900=15min)")
@click.option("--dry-run", is_flag=True, help="Run without executing actual orders")
def live(interval: int, dry_run: bool) -> None:
    """Start live trading on Hyperliquid."""
    if not CONFIG.exchange.private_key:
        console.print("[red]ERROR: HL_PRIVATE_KEY not set in .env[/red]")
        console.print("Copy .env.example to .env and fill in your credentials")
        sys.exit(1)

    if dry_run:
        console.print("[yellow]DRY RUN mode — signals computed but no orders placed[/yellow]")

    try:
        CONFIG.validate()
    except ValueError as e:
        console.print(f"[red]Config error: {e}[/red]")
        sys.exit(1)

    from strategy.master_strategy import MasterStrategy

    console.print(f"\n[green]Starting live strategy | {CONFIG.asset.symbol}/USDC | "
                  f"Capital: ${CONFIG.capital.initial_capital:,.0f}[/green]")
    console.print(f"Loop interval: {interval}s | Mainnet: {CONFIG.exchange.mainnet}\n")

    strategy = MasterStrategy(CONFIG)

    while True:
        try:
            state = strategy.run_once()

            table = Table(title=f"Strategy State — {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Portfolio Value", f"${state.get('portfolio_value', 0):,.2f}")
            table.add_row("Regime", state.get("regime", "?").upper())
            table.add_row("Composite Signal", f"{state.get('composite_signal', 0):+.3f}")
            table.add_row("Signal Strength", f"{state.get('signal_strength', 0):.3f}")
            table.add_row("Agreeing Signals", str(state.get("n_agreeing_signals", 0)))
            table.add_row("Open Trades", str(state.get("open_trades", 0)))
            table.add_row("Portfolio Heat", f"{state.get('portfolio_heat', 0):.1%}")
            table.add_row("Status", state.get("status", "?").upper())

            console.print(table)

            time.sleep(interval)

        except KeyboardInterrupt:
            console.print("\n[yellow]Strategy stopped by user[/yellow]")
            report = strategy.get_performance_report()
            analytics = PerformanceAnalytics(CONFIG.capital.initial_capital, CONFIG.capital.target_capital)
            console.print(analytics.generate_report(report))
            break

        except Exception as e:
            logger.error(f"Live trading error: {e}", exc_info=True)
            console.print(f"[red]Error: {e}[/red]")
            time.sleep(60)


@cli.command()
@click.option("--start", default=None, help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD")
@click.option("--capital", default=5000.0, help="Initial capital USD")
@click.option("--simple", is_flag=True, help="Simple backtest (no walk-forward)")
@click.option("--save-chart", default=None, help="Save HTML chart to path")
def backtest(
    start: str,
    end: str,
    capital: float,
    simple: bool,
    save_chart: str,
) -> None:
    """Run walk-forward backtest on historical data."""
    from exchange.hyperliquid_client import HyperliquidClient
    from backtesting.engine import BacktestEngine
    from backtesting.metrics import print_metrics_report

    os.environ["BACKTESTING"] = "1"  # Suppresses live trading credential check

    console.print(f"\n[cyan]Fetching HYPE/USDC historical data...[/cyan]")

    client = HyperliquidClient("", "", mainnet=True)

    start_date = start or CONFIG.backtest.start_date
    end_date = end or CONFIG.backtest.end_date

    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]")) as progress:
        task = progress.add_task("Fetching 15-minute candles...", total=None)
        df = client.get_candles(CONFIG.asset.symbol, "15m", start_ms, end_ms)
        funding_df = client.get_funding_history(CONFIG.asset.symbol, start_ms, end_ms)
        progress.update(task, completed=True)

    if df.empty:
        console.print("[red]No data fetched — check date range or network connection[/red]")
        sys.exit(1)

    console.print(f"[green]Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}[/green]")

    engine = BacktestEngine(CONFIG, initial_capital=capital)

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]")) as progress:
        task = progress.add_task("Running backtest...", total=None)
        results = engine.run(df, funding_df, walk_forward=not simple)
        progress.update(task, completed=True)

    metrics = results["metrics"]
    print_metrics_report(metrics, title=f"HYPE/USDC Backtest | {start_date} → {end_date}")

    analytics = PerformanceAnalytics(capital, CONFIG.capital.target_capital)
    console.print(analytics.generate_report(metrics))

    if save_chart:
        try:
            fig = analytics.create_dashboard(
                results["equity_curve"],
                results["trades"],
                metrics,
            )
            if fig:
                fig.write_html(save_chart)
                console.print(f"[green]Chart saved to {save_chart}[/green]")
        except Exception as e:
            console.print(f"[yellow]Chart save failed: {e}[/yellow]")


@cli.command()
def report() -> None:
    """Print compound growth schedule and strategy overview."""
    analytics = PerformanceAnalytics(
        initial_capital=CONFIG.capital.initial_capital,
        target_capital=CONFIG.capital.target_capital,
        target_months=CONFIG.capital.target_years * 12,
    )

    analytics.print_compound_schedule()

    # Print strategy overview
    console.print("\n[bold cyan]STRATEGY OVERVIEW[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=25)
    table.add_column("Description", style="white", width=45)

    rows = [
        ("Asset", "HYPE/USDC Perpetual — Hyperliquid DEX"),
        ("Primary TF", "15-minute bars"),
        ("Signal Sources", "Technical, Momentum, Mean Reversion,\nOrder Flow, Funding, Volatility, ML"),
        ("Regime Detection", "HMM (3 states) + Kalman Filter"),
        ("ML Models", "BiLSTM (128→64→32) + XGBoost (500 trees)"),
        ("Position Sizing", "Fractional Kelly (25%) + Vol Targeting"),
        ("Risk Control", "Daily/Weekly/Monthly circuit breakers"),
        ("Stop Loss", "2x ATR dynamic trailing stop"),
        ("Take Profit", "5x ATR = 2.5R reward"),
        ("Max Leverage", f"{CONFIG.asset.max_leverage}x (schedule-based)"),
        ("Kelly Fraction", f"{CONFIG.risk.kelly_fraction:.0%} of full Kelly"),
        ("Max Drawdown", f"{CONFIG.risk.max_drawdown_pct:.0%} halt trigger"),
        ("Walk-Forward", "Train 180d → Test 30d → Step 30d"),
        ("Transaction Costs", "0.02% maker / 0.05% taker + 2bps slippage"),
    ]

    for name, desc in rows:
        table.add_row(name, desc)

    console.print(table)


@cli.command()
def schedule() -> None:
    """Print month-by-month compound growth target schedule."""
    analytics = PerformanceAnalytics(
        initial_capital=CONFIG.capital.initial_capital,
        target_capital=CONFIG.capital.target_capital,
        target_months=CONFIG.capital.target_years * 12,
    )

    monthly_r = (CONFIG.capital.target_capital / CONFIG.capital.initial_capital) ** (1 / 48) - 1
    df = analytics._target_schedule

    console.print(f"\n[bold green]Compound Growth Schedule[/bold green]")
    console.print(f"[dim]$5,000 → $1,000,000 in 48 months | Monthly Return: {monthly_r:.2%}[/dim]\n")

    table = Table()
    table.add_column("Month", style="cyan", justify="right")
    table.add_column("Target Value", style="green", justify="right")
    table.add_column("Multiple", style="yellow", justify="right")
    table.add_column("Monthly Gain", style="white", justify="right")

    checkpoints = [0, 1, 3, 6, 12, 18, 24, 30, 36, 42, 48]
    for _, row in df[df["month"].isin(checkpoints)].iterrows():
        m = int(row["month"])
        val = row["portfolio_value"]
        mult = val / CONFIG.capital.initial_capital
        monthly_gain = val * monthly_r if m > 0 else 0.0
        table.add_row(
            f"Month {m:>2}",
            f"${val:>15,.2f}",
            f"{mult:>8.1f}x",
            f"${monthly_gain:>12,.2f}" if m > 0 else "—",
        )

    console.print(table)


if __name__ == "__main__":
    cli()
