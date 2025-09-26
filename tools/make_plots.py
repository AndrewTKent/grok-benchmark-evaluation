# tools/make_plots.py
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NOTE: per your constraints, we use matplotlib, one chart per figure, and no explicit colors.

def load_metrics(run_dir: Path) -> pd.DataFrame:
    """Recursively load per-task enhanced metrics JSON files."""
    rows = []
    for mf in run_dir.rglob("enhanced_metrics/metrics_*.json"):
        try:
            with mf.open(encoding="utf-8") as f:
                m = json.load(f)
            rows.append(m)
        except Exception:
            pass
    return pd.DataFrame(rows)

def load_run_summary(run_dir: Path) -> dict:
    """Try to load run-level enhanced_analysis.json / results.json if available."""
    out = {}
    for p in [run_dir / "enhanced_analysis.json", run_dir / "enhanced_analysis.txt", run_dir / "results.json"]:
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

def plot_success_time(df: pd.DataFrame, figs: Path):
    """Per-task time (bars) with success markers."""
    if df.empty: return
    df2 = df.copy()
    df2["success"] = df2["success"].astype(bool)
    df2["task_id"] = df2.get("task_id", pd.Series([f"task_{i}" for i in range(len(df2))]))
    df2 = df2.sort_values("time_taken")
    plt.figure()
    plt.bar(df2["task_id"], df2["time_taken"])
    # Success markers as dots above bars
    y = df2["time_taken"].to_numpy()
    x = np.arange(len(df2))
    succ = df2["success"].to_numpy()
    plt.scatter(x[succ], y[succ]*1.02 + 0.01, s=30)  # small dot for successes
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Time (s)")
    plt.title("Per-task time (bars) with success markers (dots)")
    plt.tight_layout()
    plt.savefig(figs / "time_per_task.png", dpi=180)
    plt.close()

def plot_component_bars(df: pd.DataFrame, figs: Path):
    """Average component scores."""
    if df.empty: return
    comps = ["composite_score", "efficiency_score", "recovery_score", "safety_score"]
    present = [c for c in comps if c in df.columns]
    if not present: return
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
    """Stacked/aggregated failure pattern counts (flat bar of totals)."""
    if df.empty: return
    total = Counter()
    for d in df.get("error_patterns", []):
        if isinstance(d, dict):
            total.update(d)
    if not total:
        # Look for violation patterns as proxy (safety_violations list of dicts)
        vio = Counter()
        for lst in df.get("safety_violations", []):
            # df["safety_violations"] can be a list-of-dicts blobs per row; normalize:
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
    if df.empty or "loop_count" not in df.columns: return
    vals = pd.to_numeric(df["loop_count"], errors="coerce").dropna()
    if len(vals) == 0: return
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
    if df.empty: return
    if not {"steps_taken", "time_taken"}.issubset(df.columns): return
    vals = df[["steps_taken", "time_taken"]].dropna()
    if vals.empty: return
    plt.figure()
    plt.scatter(vals["steps_taken"], vals["time_taken"])
    plt.xlabel("Steps taken")
    plt.ylabel("Time (s)")
    plt.title("Steps vs. time")
    plt.tight_layout()
    plt.savefig(figs / "steps_vs_time.png", dpi=180)
    plt.close()

def plot_success_rate(df: pd.DataFrame, figs: Path):
    """Simple success vs failure bar."""
    if df.empty or "success" not in df.columns: return
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

def main():
    ap = argparse.ArgumentParser(description="Generate figures from Enhanced TB metrics.")
    ap.add_argument("--run-dir", required=True, help="Path to a specific run directory (e.g., results/tb_..._YYYYMMDD_HHMMSS)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    figs = ensure_figs_dir(run_dir)
    df = load_metrics(run_dir)

    # Normalize some columns if present
    if "time_taken" in df.columns:
        df["time_taken"] = pd.to_numeric(df["time_taken"], errors="coerce")
    if "steps_taken" in df.columns:
        df["steps_taken"] = pd.to_numeric(df["steps_taken"], errors="coerce")

    # Build plots
    plot_success_rate(df, figs)
    plot_success_time(df, figs)
    plot_component_bars(df, figs)
    plot_failure_patterns(df, figs)
    plot_loops_hist(df, figs)
    plot_steps_vs_time(df, figs)

    print(f"Figures written to: {figs}")

if __name__ == "__main__":
    main()
