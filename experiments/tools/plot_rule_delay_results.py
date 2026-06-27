"""Plot ThinkAgent vs delayed Rule1 win-rate curves.

Reads CSVs produced by `Experiments/run_exp4_rule_delay.py`:
- think_vs_rule_delay_chess_*.csv
- think_vs_rule_delay_maze_*.csv

Produces line charts with:
- x-axis: delay seconds
- y-axis: win rate

By default it searches `Experiments/records/` for the newest matching CSVs.

Usage (Windows / repo root):
    python Experiments\\tools\\plot_rule_delay_results.py
    python Experiments\\tools\\plot_rule_delay_results.py --chess Experiments\\records\\think_vs_rule_delay_chess_20260115_221756.csv
    python Experiments\\tools\\plot_rule_delay_results.py --maze  Experiments\\records\\think_vs_rule_delay_maze_20260115_221756.csv

Outputs:
    Experiments/records/think_vs_rule_delay_chess_winrate.png
    Experiments/records/think_vs_rule_delay_maze_winrate.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "records")


@dataclass(frozen=True)
class Row:
    delay_s: float
    think_as: str
    episodes: int
    think_win_rate: float | None
    rule_win_rate: float | None
    draw_rate: float | None


def _safe_float(x: object) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int(x: object) -> int | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _read_delay_csv(path: str) -> list[Row]:
    rows: list[Row] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            delay_s = _safe_float(r.get("delay_s"))
            if delay_s is None:
                continue
            think_as = str(r.get("think_as") or "").strip() or "(unknown)"
            episodes = _safe_int(r.get("episodes")) or 0
            rows.append(
                Row(
                    delay_s=float(delay_s),
                    think_as=think_as,
                    episodes=int(episodes),
                    think_win_rate=_safe_float(r.get("think_win_rate")),
                    rule_win_rate=_safe_float(r.get("rule_win_rate")),
                    draw_rate=_safe_float(r.get("draw_rate")),
                )
            )
    return rows


def _latest_glob(pattern: str) -> str | None:
    paths = glob.glob(pattern)
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def _weighted_avg(points: Iterable[tuple[float, float]], weights: Iterable[int]) -> float | None:
    num = 0.0
    den = 0.0
    for (x, y), w in zip(points, weights):
        if y is None:
            continue
        ww = float(w)
        num += float(y) * ww
        den += ww
    return (num / den) if den > 0 else None


def plot_delay_winrate(*, csv_path: str, title: str, out_png: str) -> None:
    rows = _read_delay_csv(csv_path)
    if not rows:
        raise RuntimeError(f"No valid rows in CSV: {csv_path}")

    # Group by delay.
    delays = sorted({r.delay_s for r in rows})

    # Compute overall (weighted by episodes) win rates per delay.
    overall_think: list[float | None] = []
    overall_rule: list[float | None] = []
    for d in delays:
        rs = [r for r in rows if r.delay_s == d]
        think_points = [(d, r.think_win_rate) for r in rs]
        rule_points = [(d, r.rule_win_rate) for r in rs]
        w = [max(0, int(r.episodes)) for r in rs]
        overall_think.append(_weighted_avg(think_points, w))
        overall_rule.append(_weighted_avg(rule_points, w))

    # Also plot per-side curves (thin lines) if multiple sides exist.
    by_side: dict[str, dict[float, Row]] = {}
    for r in rows:
        by_side.setdefault(r.think_as, {})[r.delay_s] = r

    plt.figure(figsize=(9, 5))

    # Side curves
    side_names = sorted(by_side.keys())
    if len(side_names) > 1:
        for side in side_names:
            xs: list[float] = []
            ys: list[float] = []
            for d in delays:
                rr = by_side.get(side, {}).get(d)
                if rr is None or rr.think_win_rate is None:
                    continue
                xs.append(d)
                ys.append(float(rr.think_win_rate))
            if xs:
                plt.plot(xs, ys, linestyle="--", linewidth=1.0, alpha=0.5, label=f"Think win ({side})")

    # Overall curves
    xs_overall = [d for d, y in zip(delays, overall_think) if y is not None]
    ys_overall = [float(y) for y in overall_think if y is not None]
    if xs_overall:
        plt.plot(xs_overall, ys_overall, marker="o", linewidth=2.5, label="Think win (overall)")

    xs_rule = [d for d, y in zip(delays, overall_rule) if y is not None]
    ys_rule = [float(y) for y in overall_rule if y is not None]
    if xs_rule:
        plt.plot(xs_rule, ys_rule, marker="o", linewidth=2.5, label="Rule win (overall)")

    plt.ylim(-0.02, 1.02)
    plt.xlabel("Rule agent delay (seconds)")
    plt.ylabel("Win rate")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chess", type=str, default="", help="Path to think_vs_rule_delay_chess_*.csv")
    ap.add_argument("--maze", type=str, default="", help="Path to think_vs_rule_delay_maze_*.csv")
    ap.add_argument("--outdir", type=str, default=RESULTS_DIR, help="Output directory for PNGs")
    args = ap.parse_args()

    chess_csv = args.chess.strip() if args.chess else ""
    maze_csv = args.maze.strip() if args.maze else ""

    if not chess_csv:
        chess_csv = _latest_glob(os.path.join(RESULTS_DIR, "think_vs_rule_delay_chess_*.csv")) or ""
    if not maze_csv:
        maze_csv = _latest_glob(os.path.join(RESULTS_DIR, "think_vs_rule_delay_maze_*.csv")) or ""

    outdir = os.path.abspath(args.outdir)

    did_any = False
    if chess_csv and os.path.exists(chess_csv):
        plot_delay_winrate(
            csv_path=chess_csv,
            title="ThinkAgent vs Rule1 (delayed) — Chess",
            out_png=os.path.join(outdir, "think_vs_rule_delay_chess_winrate.png"),
        )
        print(f"Saved: {os.path.join(outdir, 'think_vs_rule_delay_chess_winrate.png')}")
        did_any = True
    else:
        print("[plot_rule_delay_results] chess CSV not found; skip")

    if maze_csv and os.path.exists(maze_csv):
        plot_delay_winrate(
            csv_path=maze_csv,
            title="ThinkAgent vs Rule1 (delayed) — Maze",
            out_png=os.path.join(outdir, "think_vs_rule_delay_maze_winrate.png"),
        )
        print(f"Saved: {os.path.join(outdir, 'think_vs_rule_delay_maze_winrate.png')}")
        did_any = True
    else:
        print("[plot_rule_delay_results] maze CSV not found; skip")

    if not did_any:
        raise SystemExit("No input CSV found. Put CSVs under Experiments/records/ or pass --chess/--maze.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
