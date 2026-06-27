from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from adjustText import adjust_text  # type: ignore
except Exception:  # pragma: no cover
    adjust_text = None


@dataclass
class AgentAgg:
    win_rates: list[float]
    decision_times: list[float]
    tokens_per_step: list[float]


def _to_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _mean(xs: list[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _latest_csv(pattern: str) -> str:
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV matched: {pattern}")
    # Filenames include timestamps; lexical sort is fine.
    return paths[-1]


def _latest_summary_csv(results_dir: str, stem: str) -> str:
    """Pick the latest *summary* CSV, excluding per-episode files."""
    pattern = os.path.join(results_dir, f"{stem}_*.csv")
    paths = sorted(glob(pattern))
    paths = [p for p in paths if "_episodes_" not in os.path.basename(p)]
    if not paths:
        raise FileNotFoundError(f"No summary CSV matched: {pattern}")
    return paths[-1]


def _read_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def aggregate_chess_main_csv(path: str) -> dict[str, AgentAgg]:
    rows = _read_rows(path)
    out: dict[str, AgentAgg] = {}

    for r in rows:
        white = (r.get("white") or "").strip()
        black = (r.get("black") or "").strip()

        w_wr = _to_float(r.get("white_win_rate"))
        b_wr = _to_float(r.get("black_win_rate"))
        w_dt = _to_float(r.get("white_avg_decision_time"))
        b_dt = _to_float(r.get("black_avg_decision_time"))
        w_tps = _to_float(r.get("white_avg_tokens_per_step"))
        b_tps = _to_float(r.get("black_avg_tokens_per_step"))
        w_steps = _to_float(r.get("white_avg_steps"))
        b_steps = _to_float(r.get("black_avg_steps"))

        # White side
        if white:
            agg = out.setdefault(white, AgentAgg([], [], []))
            if w_wr is not None:
                agg.win_rates.append(w_wr)
            # If avg_steps==0, the time/token values are usually dummy zeros; skip them.
            if (w_steps is None or w_steps > 0) and (w_dt is not None and w_dt > 0):
                agg.decision_times.append(w_dt)
            if (w_steps is None or w_steps > 0) and (w_tps is not None and w_tps > 0):
                agg.tokens_per_step.append(w_tps)

        # Black side
        if black:
            agg = out.setdefault(black, AgentAgg([], [], []))
            if b_wr is not None:
                agg.win_rates.append(b_wr)
            if (b_steps is None or b_steps > 0) and (b_dt is not None and b_dt > 0):
                agg.decision_times.append(b_dt)
            if (b_steps is None or b_steps > 0) and (b_tps is not None and b_tps > 0):
                agg.tokens_per_step.append(b_tps)

    return out


def aggregate_maze_main_csv(path: str) -> dict[str, AgentAgg]:
    rows = _read_rows(path)
    out: dict[str, AgentAgg] = {}

    for r in rows:
        red = (r.get("red") or "").strip()
        blue = (r.get("blue") or "").strip()

        r_wr = _to_float(r.get("red_win_rate"))
        c_wr = _to_float(r.get("blue_win_rate"))
        r_dt = _to_float(r.get("red_avg_decision_time"))
        c_dt = _to_float(r.get("blue_avg_decision_time"))
        r_tps = _to_float(r.get("red_avg_tokens_per_step"))
        c_tps = _to_float(r.get("blue_avg_tokens_per_step"))
        r_steps = _to_float(r.get("red_avg_steps"))
        c_steps = _to_float(r.get("blue_avg_steps"))

        if red:
            agg = out.setdefault(red, AgentAgg([], [], []))
            if r_wr is not None:
                agg.win_rates.append(r_wr)
            if (r_steps is None or r_steps > 0) and (r_dt is not None and r_dt > 0):
                agg.decision_times.append(r_dt)
            if (r_steps is None or r_steps > 0) and (r_tps is not None and r_tps > 0):
                agg.tokens_per_step.append(r_tps)

        if blue:
            agg = out.setdefault(blue, AgentAgg([], [], []))
            if c_wr is not None:
                agg.win_rates.append(c_wr)
            if (c_steps is None or c_steps > 0) and (c_dt is not None and c_dt > 0):
                agg.decision_times.append(c_dt)
            if (c_steps is None or c_steps > 0) and (c_tps is not None and c_tps > 0):
                agg.tokens_per_step.append(c_tps)

    return out


def _assign_agent_colors(agent_names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    return _assign_agent_colors_with_palette(agent_names, palette="tab10")


def _assign_agent_colors_with_palette(
    agent_names: list[str],
    palette: str,
) -> dict[str, tuple[float, float, float, float]]:
    """Stable mapping from agent name -> RGBA color.

    palette examples: tab10, tab20, Set2, Set3.
    """
    names = sorted(set(n for n in agent_names if n))
    if not names:
        return {}

    cmap = plt.get_cmap(palette)
    color_list = getattr(cmap, "colors", None)
    if color_list:
        return {name: color_list[i % len(color_list)] for i, name in enumerate(names)}

    # Continuous colormap fallback.
    denom = max(1, len(names) - 1)
    return {name: cmap(i / denom) for i, name in enumerate(names)}


def _build_size(tokens_per_step: Optional[float], max_tokens: float) -> float:
    base = 140.0
    span = 1600.0
    if tokens_per_step is None or tokens_per_step <= 0:
        return base
    return base + span * math.sqrt(float(tokens_per_step) / float(max_tokens))


def _compute_points(stats: dict[str, AgentAgg]) -> list[tuple[str, float, float, Optional[float]]]:
    points: list[tuple[str, float, float, Optional[float]]] = []
    for name, agg in stats.items():
        x = _mean(agg.decision_times)
        y = _mean(agg.win_rates)
        s = _mean(agg.tokens_per_step)
        if x is None or y is None:
            continue
        points.append((name, float(x), float(y), s if s is None else float(s)))
    return points


def _pareto_front(points: list[tuple[str, float, float, Optional[float]]]) -> list[tuple[str, float, float]]:
    """Maximize y (win rate), minimize x (time)."""
    pts = sorted(points, key=lambda p: (p[1], -p[2]))
    front: list[tuple[str, float, float]] = []
    best_y = -1.0
    for name, x, y, _s in pts:
        if y > best_y:
            front.append((name, x, y))
            best_y = y
    return front


def _scatter(
    ax,
    stats: dict[str, AgentAgg],
    title: str,
    agent_colors: dict[str, tuple[float, float, float, float]],
    label_mode: str = "auto",
    topk: int = 10,
    xlim: Optional[tuple[float, float]] = None,
    xscale: str = "linear",
    show_pareto: bool = True,
    use_adjust_text: bool = True,
):
    points = _compute_points(stats)

    if not points:
        ax.set_title(title + " (no data)")
        return

    max_tokens = max((p[3] for p in points if p[3] is not None), default=0.0) or 1.0

    xs, ys, sizes, colors = [], [], [], []
    for name, x, y, tokens in points:
        xs.append(x)
        ys.append(y)
        sizes.append(_build_size(tokens, max_tokens))
        colors.append(agent_colors.get(name, (0.2, 0.2, 0.2, 1.0)))

    ax.scatter(
        xs,
        ys,
        s=sizes,
        c=colors,
        alpha=0.60,
        edgecolors="white",
        linewidths=0.9,
        zorder=3,
    )

    if show_pareto and len(points) >= 2:
        front = _pareto_front(points)
        if len(front) >= 2:
            fx = [p[1] for p in front]
            fy = [p[2] for p in front]
            ax.plot(fx, fy, color="#222222", linewidth=1.6, alpha=0.9, zorder=2, label="Pareto front")

    # Labels: avoid turning the figure into a word cloud.
    selected = points
    if label_mode == "none":
        selected = []
    elif label_mode == "topk":
        # Prefer high win-rate points.
        selected = sorted(points, key=lambda p: (p[2], -(p[1] or 0.0)), reverse=True)[: max(1, topk)]
    elif label_mode == "auto":
        if len(points) > topk:
            selected = sorted(points, key=lambda p: (p[2], -(p[1] or 0.0)), reverse=True)[: max(1, topk)]

    texts = []
    for name, x, y, _tokens in selected:
        texts.append(
            ax.text(
                x,
                y,
                name,
                fontsize=9,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                zorder=4,
            )
        )

    if texts and use_adjust_text and adjust_text is not None:
        try:
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7, alpha=0.8),
                expand_text=(1.05, 1.18),
                expand_points=(1.12, 1.20),
            )
        except Exception:
            pass

    ax.set_title(title)
    ax.set_xlabel("avg decision time per step (s)")
    ax.set_ylabel("avg win rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    if xscale in {"linear", "log"}:
        ax.set_xscale(xscale)
    if xlim is not None:
        ax.set_xlim(xlim)

    # Subtle styling.
    ax.set_facecolor("#fbfbfb")
    for spine in ax.spines.values():
        spine.set_alpha(0.35)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(root, "records")
    default_chess = _latest_summary_csv(results_dir, "main_chess")
    default_maze = _latest_summary_csv(results_dir, "main_maze")

    ap = argparse.ArgumentParser(description="Agent scatter plot: time vs winrate, size=tokens/step (Chess+Maze).")
    ap.add_argument("--chess", default=default_chess, help="Path to main_chess_*.csv (summary CSV)")
    ap.add_argument("--maze", default=default_maze, help="Path to main_maze_*.csv (summary CSV)")
    ap.add_argument("--out", default="", help="Output PNG path (default: Experiments/records/agent_scatter_<ts>.png)")
    ap.add_argument(
        "--labels",
        default="auto",
        choices=["auto", "none", "topk", "all"],
        help="Point label mode (auto=topk when many points)",
    )
    ap.add_argument("--topk", type=int, default=10, help="If labels=auto/topk: number of labels")
    ap.add_argument(
        "--palette",
        default="tab10",
        choices=["tab10", "tab20", "Set2", "Set3"],
        help="High-contrast categorical palette",
    )
    ap.add_argument(
        "--xscale",
        default="linear",
        choices=["linear", "log"],
        help="X axis scale (log can help when times vary widely)",
    )
    ap.add_argument(
        "--unify-x",
        action="store_true",
        help="Unify X-axis limits across Chess/Maze for easier comparison",
    )
    ap.add_argument(
        "--no-pareto",
        action="store_true",
        help="Disable Pareto front overlay",
    )
    ap.add_argument(
        "--no-adjust-text",
        action="store_true",
        help="Disable adjustText label de-overlap (useful if not installed)",
    )
    args = ap.parse_args()

    chess_stats = aggregate_chess_main_csv(args.chess)
    maze_stats = aggregate_maze_main_csv(args.maze)

    all_agents = list(chess_stats.keys()) + list(maze_stats.keys())
    agent_colors = _assign_agent_colors_with_palette(all_agents, palette=args.palette)

    # Global style: more readable defaults.
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    # Layout: top row has the two plots; bottom row is reserved for legends and notes.
    fig = plt.figure(figsize=(16.8, 7.8))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 0.20], hspace=0.28, wspace=0.18)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis("off")

    label_mode = args.labels
    if label_mode == "all":
        label_mode = "auto"  # handled by setting topk very large below
        topk = 10**9
    else:
        topk = max(1, int(args.topk))

    chess_points = _compute_points(chess_stats)
    maze_points = _compute_points(maze_stats)

    xlim = None
    if args.unify_x:
        all_points = chess_points + maze_points
        if all_points:
            xs = [p[1] for p in all_points if p[1] is not None and p[1] > 0]
            if xs:
                lo = min(xs)
                hi = max(xs)
                if args.xscale == "log":
                    lo = max(lo * 0.85, 1e-3)
                    hi = hi * 1.15
                else:
                    pad = 0.08 * (hi - lo) if hi > lo else max(0.5, 0.08 * hi)
                    lo = max(0.0, lo - pad)
                    hi = hi + pad
                xlim = (lo, hi)

    _scatter(
        axes[0],
        chess_stats,
        title="Chess",
        agent_colors=agent_colors,
        label_mode=label_mode,
        topk=topk,
        xlim=xlim,
        xscale=args.xscale,
        show_pareto=not args.no_pareto,
        use_adjust_text=not args.no_adjust_text,
    )
    _scatter(
        axes[1],
        maze_stats,
        title="Maze",
        agent_colors=agent_colors,
        label_mode=label_mode,
        topk=topk,
        xlim=xlim,
        xscale=args.xscale,
        show_pareto=not args.no_pareto,
        use_adjust_text=not args.no_adjust_text,
    )

    out_path = args.out.strip()
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(root, "records", f"agent_scatter_{ts}.png")

    fig.patch.set_facecolor("white")
    fig.suptitle("Agents: decision time vs win rate (bubble area = tokens/step)", fontsize=16, y=0.985)

    # Shared legends + notes in the dedicated bottom area.
    legend_names = sorted(agent_colors.keys())
    if legend_names:
        agent_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=agent_colors[n],
                markeredgecolor="white",
                markersize=9,
            )
            for n in legend_names
        ]
        leg1 = ax_legend.legend(
            agent_handles,
            legend_names,
            loc="center left",
            ncol=min(6, max(1, len(legend_names))),
            frameon=True,
            fontsize=9,
            title="Agent colors",
            title_fontsize=10,
        )
        leg1.get_frame().set_alpha(0.95)
        ax_legend.add_artist(leg1)

    # One bubble-size legend (reference only; NOT an agent).
    global_max_tokens = max(
        max([p[3] for p in chess_points if p[3] is not None and p[3] > 0], default=0.0),
        max([p[3] for p in maze_points if p[3] is not None and p[3] > 0], default=0.0),
    ) or 1.0
    refs = [0.25 * global_max_tokens, 0.5 * global_max_tokens, 1.0 * global_max_tokens]
    size_handles = [
        ax_legend.scatter(
            [],
            [],
            s=_build_size(v, global_max_tokens),
            facecolors="none",
            edgecolors="#666666",
            linewidths=1.1,
        )
        for v in refs
    ]
    size_labels = [f"{v:.0f} tok/step" for v in refs]
    leg2 = ax_legend.legend(
        size_handles,
        size_labels,
        title="Bubble size (reference, not an agent)",
        title_fontsize=10,
        loc="center right",
        frameon=True,
        fontsize=9,
    )
    leg2.get_frame().set_alpha(0.95)

    if (not args.no_adjust_text) and adjust_text is None:
        ax_legend.text(
            0.5,
            0.06,
            "Tip: install adjustText to automatically fix label overlap: pip install adjustText",
            ha="center",
            va="center",
            fontsize=9,
            color="#444444",
            transform=ax_legend.transAxes,
        )

    # Sources note
    ax_legend.text(
        0.5,
        -0.12,
        f"Sources: {os.path.basename(args.chess)} | {os.path.basename(args.maze)}",
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
        transform=ax_legend.transAxes,
    )
    fig.savefig(out_path, dpi=200)
    print(f"Chess CSV: {args.chess}")
    print(f"Maze  CSV: {args.maze}")
    print(f"Saved: {out_path}")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
