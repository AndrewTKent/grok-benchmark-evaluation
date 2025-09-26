#!/usr/bin/env python3
"""
tools/make_plots.py

Generate figures from Enhanced Terminal-Bench per-task metrics.
Also supports comparing a baseline results.json vs an enhanced enhanced_analysis.json.

Usage (single run, as before):
    python tools/make_plots.py --run-dir results/tb_grok-4-fast-reasoning_enhanced_20250926_051857

Usage (comparison):
    python tools/make_plots.py \
      --baseline-results results/tb_grok-4-fast-reasoning_20250926_051255/2025-09-26__05-12-59/results.json \
      --enhanced-analysis results/tb_grok-4-fast-reasoning_enhanced_20250926_051857/enhanced_analysis.json \
      [--enhanced-run-dir results/tb_grok-4-fast-reasoning_enhanced_20250926_051857] \
      [--out-dir results/figs_compare]

Outputs (single run):
    <run-dir>/figs/
        success_rate.png
        component_scores.png
        failure_patterns.png
        loops_hist.png
        steps_vs_time.png
        time_ecdf.png
        time_hist.png
        time_hist_log.png
        time_box.png
        time_box_by_outcome.png
        time_fastest_topN.png
        time_slowest_bottomN.png

Outputs (comparison):
    <out-dir>/comp_success_rate.png
    <out-dir>/comp_component_scores.png
    (future comparisons will be added only if both sides have the needed fields)

Notes:
- Uses matplotlib only, one chart per figure, no explicit colors.
- Robust to missing fields; silently skips plots if required columns absent.
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------- IO / Loading (single-run helpers) ----------

def load_metrics(run_dir: Path) -> pd.DataFrame:
    """Recursively load per-task enhanced metrics JSON files."""
    rows = []
    for mf in run_dir.rglob("enhanced_metrics/metrics_*.json"):
        try:
            with mf.open(encoding="utf-8") as f:
                m = json.load(f)
            rows.append(m)
        except Exception:
            # Skip any unreadable/corrupt file
            pass
    return pd.DataFrame(rows)


def load_run_summary(run_dir: Path) -> dict:
    """Try to load run-level JSON summaries if available."""
    out = {}
    for p in [run_dir / "enhanced_analysis.json", run_dir / "results.json"]:
        if p.exists() and p.suffix.lower() == ".json":
            try:
                out[p.name] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return out


def ensure_figs_dir(run_dir: Path) -> Path:
    figs = run_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    return figs


# ---------- IO / Loading (comparison helpers) ----------

def parse_baseline_results_json(path: Path) -> dict:
    """
    Parse a baseline results.json to extract success counts and (if present) task durations.
    Returns dict with keys: succ, fail, total, success_rate, times (list|None)
    """
    if not path.exists():
        raise FileNotFoundError(f"Baseline results not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to parse baseline results: {e}")

    succ = 0
    total = 0
    times = []

    # Common schema from tb_runner_base aggregation is a run-level "tasks" list.
    tasks = data.get("tasks")
    if isinstance(tasks, list):
        for t in tasks:
            status = str(t.get("status", "")).lower()
            if status:
                total += 1
                if status == "success":
                    succ += 1
            # Try to capture any timing if exposed
            # (not guaranteed; many results.json don't include per-task times)
            tt = t.get("time_taken") or t.get("duration") or None
            if isinstance(tt, (int, float)):
                times.append(float(tt))

    # Fallback: try a nested field (some schemas differ)
    if total == 0 and isinstance(data, dict):
        # Look for trial-like objects
        for k, v in data.items():
            if isinstance(v, dict) and "status" in v:
                total += 1
                succ += 1 if str(v["status"]).lower() == "success" else 0

    fail = max(0, total - succ)
    rate = (succ / total) if total else 0.0
    return {"succ": succ, "fail": fail, "total": total, "success_rate": rate, "times": times or None}


def parse_enhanced_analysis_json(path: Path) -> dict:
    """
    Parse enhanced_analysis.json to extract summary metrics.
    Returns dict with keys: success_rate, composite, efficiency, recovery, safety
    Missing keys are set to None.
    """
    if not path.exists():
        raise FileNotFoundError(f"Enhanced analysis not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to parse enhanced analysis: {e}")

    summary = data.get("analysis", {}).get("summary", {}) if "analysis" in data else data.get("summary", {})
    return {
        "success_rate": summary.get("success_rate"),
        "composite": summary.get("avg_composite_score"),
        "efficiency": summary.get("avg_efficiency"),
        "recovery": summary.get("avg_recovery"),
        "safety": summary.get("avg_safety"),
    }


def ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------- Plots (single-run) ----------

def plot_success_rate(df: pd.DataFrame, figs: Path):
    """Simple success vs failure bar."""
    if df.empty or "success" not in df.columns:
        return
    succ = int(df["success"].astype(bool).sum())
    fail = int((~df["success"].astype(bool)).sum())
    plt.figure()
    plt.bar(["Success", "Failure"], [succ, fail])
    plt.ylabel("Count")
    rate = succ / max(1, succ + fail)
    plt.title(f"Success rate: {rate:.0%}")
    plt.tight_layout()
    plt.savefig(figs / "success_rate.png", dpi=180)
    plt.close()


def plot_component_bars(df: pd.DataFrame, figs: Path):
    """Average component scores."""
    if df.empty:
        return
    comps = ["composite_score", "efficiency_score", "recovery_score", "safety_score"]
    present = [c for c in comps if c in df.columns]
    if not present:
        return
    means = df[present].mean().sort_values(ascending=False)
    plt.figure()
    plt.bar(means.index, means.values)
    plt.ylim(0, 1)
    plt.ylabel("Average score")
    plt.title("Average component scores")
    plt.tight_layout()
    plt.savefig(figs / "component_scores.png", dpi=180)
    plt.close()


def plot_failure_patterns(df: pd.DataFrame, figs: Path):
    """Aggregated failure / violation pattern counts."""
    if df.empty:
        return

    # error_patterns is a dict per row -> aggregate
    total = Counter()
    if "error_patterns" in df.columns:
        for d in df["error_patterns"]:
            if isinstance(d, dict):
                total.update(d)

    # If none, try safety_violations list-of-dicts with 'pattern'
    if not total and "safety_violations" in df.columns:
        vio = Counter()
        for lst in df["safety_violations"]:
            if isinstance(lst, list):
                for it in lst:
                    if isinstance(it, dict) and "pattern" in it:
                        vio[it["pattern"]] += 1
        total = vio

    if not total:
        return

    keys, vals = zip(*sorted(total.items(), key=lambda x: x[1], reverse=True))
    plt.figure()
    plt.bar(keys, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Failure / Violation pattern counts")
    plt.tight_layout()
    plt.savefig(figs / "failure_patterns.png", dpi=180)
    plt.close()


def plot_loops_hist(df: pd.DataFrame, figs: Path):
    """Histogram of loop counts."""
    if df.empty or "loop_count" not in df.columns:
        return
    vals = pd.to_numeric(df["loop_count"], errors="coerce").dropna()
    if len(vals) == 0:
        return
    plt.figure()
    plt.hist(vals, bins=min(10, max(3, int(vals.max()) + 1)))
    plt.xlabel("Loop count")
    plt.ylabel("Tasks")
    plt.title("Distribution of loop counts")
    plt.tight_layout()
    plt.savefig(figs / "loops_hist.png", dpi=180)
    plt.close()


def plot_steps_vs_time(df: pd.DataFrame, figs: Path):
    """Scatter: steps vs time."""
    if df.empty or not {"steps_taken", "time_taken"}.issubset(df.columns):
        return
    vals = df[["steps_taken", "time_taken"]].dropna()
    if vals.empty:
        return
    plt.figure()
    plt.scatter(vals["steps_taken"], vals["time_taken"])
    plt.xlabel("Steps taken")
    plt.ylabel("Time (s)")
    plt.title("Steps vs. time")
    plt.tight_layout()
    plt.savefig(figs / "steps_vs_time.png", dpi=180)
    plt.close()


# ---------- Scalable time visualizations (single-run) ----------

def plot_time_top_bottom(df: pd.DataFrame, figs: Path, top_n: int = 15, bottom_n: int = 15):
    """Horizontal bars for the fastest and slowest tasks (top/bottom N)."""
    if df.empty or "time_taken" not in df.columns:
        return
    d = df.copy()
    if "task_id" not in d.columns:
        d["task_id"] = [f"task_{i}" for i in range(len(d))]
    d = d[~d["time_taken"].isna()]
    if d.empty:
        return

    d_sorted = d.sort_values("time_taken")
    fastest = d_sorted.head(top_n)
    slowest = d_sorted.tail(bottom_n)

    # Fastest
    plt.figure()
    plt.barh(fastest["task_id"], fastest["time_taken"])
    plt.xlabel("Time (s)")
    plt.title(f"Fastest {len(fastest)} tasks")
    plt.tight_layout()
    plt.savefig(figs / "time_fastest_topN.png", dpi=180)
    plt.close()

    # Slowest
    plt.figure()
    plt.barh(slowest["task_id"], slowest["time_taken"])
    plt.xlabel("Time (s)")
    plt.title(f"Slowest {len(slowest)} tasks")
    plt.tight_layout()
    plt.savefig(figs / "time_slowest_bottomN.png", dpi=180)
    plt.close()


def plot_time_ecdf(df: pd.DataFrame, figs: Path):
    """ECDF of task durations (robust for large N)."""
    if df.empty or "time_taken" not in df.columns:
        return
    x = pd.to_numeric(df["time_taken"], errors="coerce").dropna().sort_values().to_numpy()
    if x.size == 0:
        return
    y = np.arange(1, x.size + 1) / x.size
    plt.figure()
    plt.plot(x, y, linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Fraction of tasks ≤ t")
    plt.title("ECDF of task durations")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(figs / "time_ecdf.png", dpi=180)
    plt.close()


def plot_time_hist_log(df: pd.DataFrame, figs: Path):
    """Histogram of durations with optional log x-scale for long tails."""
    if df.empty or "time_taken" not in df.columns:
        return
    x = pd.to_numeric(df["time_taken"], errors="coerce").dropna()
    if x.empty:
        return

    # Linear histogram
    plt.figure()
    plt.hist(x, bins=min(30, max(10, int(np.sqrt(len(x))))))
    plt.xlabel("Time (s)")
    plt.ylabel("Tasks")
    plt.title("Distribution of task durations")
    plt.tight_layout()
    plt.savefig(figs / "time_hist.png", dpi=180)
    plt.close()

    # Log-scale variant (only if strictly positive)
    xp = x[x > 0]
    if not xp.empty:
        plt.figure()
        plt.hist(xp, bins=min(30, max(10, int(np.sqrt(len(xp))))))
        plt.xscale("log")
        plt.xlabel("Time (s) [log scale]")
        plt.ylabel("Tasks")
        plt.title("Distribution of task durations (log scale)")
        plt.tight_layout()
        plt.savefig(figs / "time_hist_log.png", dpi=180)
        plt.close()


def plot_time_box(df: pd.DataFrame, figs: Path):
    """Summary five-number view for durations; add success split if available."""
    if df.empty or "time_taken" not in df.columns:
        return
    x = pd.to_numeric(df["time_taken"], errors="coerce").dropna()
    if x.empty:
        return

    # Overall boxplot
    plt.figure()
    plt.boxplot([x], vert=True, labels=["All tasks"], showfliers=False)
    plt.ylabel("Time (s)")
    plt.title("Task duration summary (no outliers shown)")
    plt.tight_layout()
    plt.savefig(figs / "time_box.png", dpi=180)
    plt.close()

    # Split by success (if present and both groups exist)
    if "success" in df.columns:
        a = pd.to_numeric(df.loc[df["success"].astype(bool), "time_taken"], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[~df["success"].astype(bool), "time_taken"], errors="coerce").dropna()
        if len(a) and len(b):
            plt.figure()
            plt.boxplot([a, b], vert=True, labels=["Success", "Failure"], showfliers=False)
            plt.ylabel("Time (s)")
            plt.title("Task duration by outcome (no outliers)")
            plt.tight_layout()
            plt.savefig(figs / "time_box_by_outcome.png", dpi=180)
            plt.close()


# ---------- Plots (comparison) ----------

def plot_comp_success_rate(baseline: dict, enhanced: dict, out_dir: Path):
    """
    Two bars: baseline vs enhanced success rate.
    baseline: dict from parse_baseline_results_json
    enhanced: dict from parse_enhanced_analysis_json
    """
    br = baseline.get("success_rate")
    er = enhanced.get("success_rate")
    if br is None and er is None:
        return
    labels, vals = [], []
    if br is not None:
        labels.append("Baseline")
        vals.append(br)
    if er is not None:
        labels.append("Enhanced")
        vals.append(er)
    if not vals:
        return
    plt.figure()
    plt.bar(labels, vals)
    plt.ylim(0, 1)
    title_bits = []
    if baseline.get("total"):
        title_bits.append(f"Baseline n={baseline['total']}")
    plt.ylabel("Success rate")
    plt.title("; ".join(title_bits) if title_bits else "Success rate comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "comp_success_rate.png", dpi=180)
    plt.close()


def plot_comp_component_scores(enhanced: dict, out_dir: Path):
    """
    Enhanced components only (baseline has no components natively).
    Bars: composite, efficiency, recovery, safety for the enhanced run.
    """
    keys = ["composite", "efficiency", "recovery", "safety"]
    present = {k: enhanced.get(k) for k in keys if enhanced.get(k) is not None}
    if not present:
        return
    names = list(present.keys())
    vals = [present[k] for k in names]
    plt.figure()
    plt.bar(names, vals)
    plt.ylim(0, 1)
    plt.ylabel("Average score")
    plt.title("Enhanced component scores")
    plt.tight_layout()
    plt.savefig(out_dir / "comp_component_scores.png", dpi=180)
    plt.close()


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Generate figures from Enhanced TB metrics or compare baseline vs enhanced.")
    # Single-run mode (original)
    ap.add_argument("--run-dir", help="Path to a run directory (e.g., results/tb_..._YYYYMMDD_HHMMSS)")
    # Comparison mode (new)
    ap.add_argument("--baseline-results", help="Path to baseline results.json")
    ap.add_argument("--enhanced-analysis", help="Path to enhanced_analysis.json")
    ap.add_argument("--enhanced-run-dir", help="(Optional) Enhanced run dir to load per-task metrics in addition to analysis.json")
    ap.add_argument("--out-dir", help="(Optional) Output dir for comparison figs (default: <enhanced-run-dir>/figs_compare or sibling of enhanced_analysis.json)")

    args = ap.parse_args()

    # --- Comparison mode if both files provided ---
    if args.baseline_results and args.enhanced_analysis:
        baseline_path = Path(args.baseline_results).resolve()
        enhanced_path = Path(args.enhanced_analysis).resolve()

        baseline = parse_baseline_results_json(baseline_path)
        enhanced = parse_enhanced_analysis_json(enhanced_path)

        # Decide output dir
        if args.out_dir:
            out_dir = ensure_out_dir(Path(args.out_dir).resolve())
        else:
            if args.enhanced_run_dir:
                out_dir = ensure_out_dir(Path(args.enhanced_run_dir).resolve() / "figs_compare")
            else:
                out_dir = ensure_out_dir(enhanced_path.parent / "figs_compare")

        # Core comparisons
        plot_comp_success_rate(baseline, enhanced, out_dir)
        plot_comp_component_scores(enhanced, out_dir)

        # Optional: if an enhanced run-dir is provided, also emit the single-run figs for that enhanced dir
        if args.enhanced_run_dir:
            run_dir = Path(args.enhanced_run_dir).resolve()
            df = load_metrics(run_dir)
            figs = ensure_figs_dir(run_dir)
            if "time_taken" in df.columns:
                df["time_taken"] = pd.to_numeric(df["time_taken"], errors="coerce")
            if "steps_taken" in df.columns:
                df["steps_taken"] = pd.to_numeric(df["steps_taken"], errors="coerce")
            plot_success_rate(df, figs)
            plot_component_bars(df, figs)
            plot_failure_patterns(df, figs)
            plot_loops_hist(df, figs)
            plot_steps_vs_time(df, figs)
            plot_time_top_bottom(df, figs, top_n=15, bottom_n=15)
            plot_time_ecdf(df, figs)
            plot_time_hist_log(df, figs)
            plot_time_box(df, figs)

        print(f"Comparison figures written to: {out_dir}")
        return

    # --- Single-run mode (original behavior) ---
    if not args.run_dir:
        raise SystemExit("Provide either --run-dir (single-run) OR both --baseline-results and --enhanced-analysis (comparison).")

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    figs = ensure_figs_dir(run_dir)
    df = load_metrics(run_dir)

    # Normalize columns when present
    if "time_taken" in df.columns:
        df["time_taken"] = pd.to_numeric(df["time_taken"], errors="coerce")
    if "steps_taken" in df.columns:
        df["steps_taken"] = pd.to_numeric(df["steps_taken"], errors="coerce")

    # Core plots
    plot_success_rate(df, figs)
    plot_component_bars(df, figs)
    plot_failure_patterns(df, figs)
    plot_loops_hist(df, figs)
    plot_steps_vs_time(df, figs)

    # Scalable time visualizations
    plot_time_top_bottom(df, figs, top_n=15, bottom_n=15)
    plot_time_ecdf(df, figs)
    plot_time_hist_log(df, figs)
    plot_time_box(df, figs)

    print(f"Figures written to: {figs}")


if __name__ == "__main__":
    main()
