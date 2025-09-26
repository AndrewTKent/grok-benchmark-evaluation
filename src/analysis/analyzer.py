# src/analysis/analyzer.py
"""Enhanced results analyzer for Terminal-Bench runs (refactored)."""

from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict
import json

import numpy as np
import pandas as pd


class EnhancedAnalyzer:
    """Analyzes enhanced metrics and generates insights."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.enhanced_metrics_dir = self.results_dir / "enhanced_metrics"
        self.metrics_data: List[Dict[str, Any]] = []
        self.load_metrics()

    def load_metrics(self) -> None:
        """Load all enhanced metrics files in the run directory."""
        if not self.enhanced_metrics_dir.exists():
            return
        for mf in self.enhanced_metrics_dir.glob("metrics_*.json"):
            try:
                with mf.open() as f:
                    self.metrics_data.append(json.load(f))
            except Exception as e:
                print(f"Error loading {mf}: {e}")

    def analyze(self) -> Dict[str, Any]:
        """Return a composite analysis dictionary."""
        if not self.metrics_data:
            return {"error": "No metrics data available"}

        return {
            "summary": self._compute_summary(),
            "failure_patterns": self._analyze_failures(),
            "loop_analysis": self._analyze_loops(),
            "recovery_analysis": self._analyze_recovery(),
            "safety_analysis": self._analyze_safety(),
            "efficiency_analysis": self._analyze_efficiency(),
            "recommendations": self._generate_recommendations(),
        }

    # ----- sub-analyses -----

    def _compute_summary(self) -> Dict[str, Any]:
        df = pd.DataFrame(self.metrics_data)
        return {
            "total_tasks": len(df),
            "success_rate": float(df["success"].mean()) if "success" in df else 0.0,
            "avg_composite_score": float(df["composite_score"].mean()) if "composite_score" in df else 0.0,
            "avg_efficiency": float(df["efficiency_score"].mean()) if "efficiency_score" in df else 0.0,
            "avg_recovery": float(df["recovery_score"].mean()) if "recovery_score" in df else 0.0,
            "avg_safety": float(df["safety_score"].mean()) if "safety_score" in df else 0.0,
            "total_loops": int(df["loop_count"].sum()) if "loop_count" in df else 0,
            "avg_steps": float(df["steps_taken"].mean()) if "steps_taken" in df else 0.0,
            "avg_time": float(df["time_taken"].mean()) if "time_taken" in df else 0.0,
        }

    def _analyze_failures(self) -> Dict[str, Any]:
        counts = defaultdict(int)
        for m in self.metrics_data:
            if not m.get("success", False):
                for k, v in (m.get("error_patterns") or {}).items():
                    counts[k] += int(v)
        sorted_errors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return {
            "most_common_errors": sorted_errors[:5],
            "total_error_types": len(counts),
            "error_distribution": dict(sorted_errors),
        }

    def _analyze_loops(self) -> Dict[str, Any]:
        loop_tasks = []
        total_loops = 0
        for m in self.metrics_data:
            loops = int(m.get("loop_count", 0))
            if loops > 0:
                loop_tasks.append({
                    "task_id": m.get("task_id", "unknown"),
                    "loops": loops,
                    "success": bool(m.get("success", False)),
                })
                total_loops += loops
        return {
            "total_loops_detected": total_loops,
            "tasks_with_loops": len(loop_tasks),
            "loop_success_rate": (sum(1 for t in loop_tasks if t["success"]) / len(loop_tasks)) if loop_tasks else 0.0,
            "worst_offenders": sorted(loop_tasks, key=lambda x: x["loops"], reverse=True)[:5],
        }

    def _analyze_recovery(self) -> Dict[str, Any]:
        scores = []
        injected = []
        for m in self.metrics_data:
            scores.append(float(m.get("recovery_score", 0.0)))
            if "injected_failures" in m:
                injected.extend(m["injected_failures"])
        return {
            "avg_recovery_score": float(np.mean(scores)) if scores else 0.0,
            "min_recovery_score": float(np.min(scores)) if scores else 0.0,
            "max_recovery_score": float(np.max(scores)) if scores else 0.0,
            "total_injected_failures": len(injected),
            "recovery_std_dev": float(np.std(scores)) if scores else 0.0,
        }

    def _analyze_safety(self) -> Dict[str, Any]:
        violations = []
        safety_scores = []
        for m in self.metrics_data:
            safety_scores.append(float(m.get("safety_score", 1.0)))
            vs = m.get("safety_violations", [])
            if isinstance(vs, list):
                violations.extend(vs)
        patterns = defaultdict(int)
        for v in violations:
            if isinstance(v, dict):
                patterns[v.get("pattern", "unknown")] += 1
        return {
            "total_violations": len(violations),
            "avg_safety_score": float(np.mean(safety_scores)) if safety_scores else 1.0,
            "violation_patterns": dict(patterns),
            "high_risk_tasks": int(sum(1 for s in safety_scores if s < 0.5)),
        }

    def _analyze_efficiency(self) -> Dict[str, Any]:
        df = pd.DataFrame(self.metrics_data)
        if df.empty:
            return {}
        out: Dict[str, Any] = {"fastest_tasks": [], "slowest_tasks": [], "most_efficient": [], "least_efficient": []}

        if "time_taken" in df.columns:
            s = df.sort_values("time_taken")
            out["fastest_tasks"] = [{"task": r.get("task_id", "unknown"), "time": float(r["time_taken"])} for _, r in s.head(3).iterrows()]
            out["slowest_tasks"] = [{"task": r.get("task_id", "unknown"), "time": float(r["time_taken"])} for _, r in s.tail(3).iterrows()]

        if "efficiency_score" in df.columns:
            s = df.sort_values("efficiency_score", ascending=False)
            out["most_efficient"] = [{"task": r.get("task_id", "unknown"), "score": float(r["efficiency_score"])} for _, r in s.head(3).iterrows()]
            out["least_efficient"] = [{"task": r.get("task_id", "unknown"), "score": float(r["efficiency_score"])} for _, r in s.tail(3).iterrows()]

        return out

    # ----- recommendations -----

    def _generate_recommendations(self) -> List[str]:
        recs: List[str] = []
        summary = self._compute_summary()
        failures = self._analyze_failures()
        loops = self._analyze_loops()

        if summary["success_rate"] < 0.5:
            recs.append("Critical: Success rate below 50%. Focus on basic task completion.")
        if loops["total_loops_detected"] > len(self.metrics_data):
            recs.append("High loop frequency detected. Improve command variation strategies.")
        if failures["most_common_errors"]:
            top, _ = failures["most_common_errors"][0]
            if top == "command_not_found":
                recs.append("Frequent 'command not found' errors. Add package installation logic.")
            elif top == "permission_denied":
                recs.append("Permission issues common. Improve sudo/chmod handling.")
            elif top == "file_not_found":
                recs.append("File path issues detected. Add existence checks before operations.")
        if summary["avg_efficiency"] < 0.5:
            recs.append("Low efficiency scores. Optimize command sequences and reduce steps.")
        if summary["avg_safety"] < 0.8:
            recs.append("Safety concerns detected. Review dangerous command patterns.")
        if summary["avg_recovery"] < 0.6:
            recs.append("Poor error recovery. Implement better fallback strategies.")

        return recs

    # ----- report writers -----

    def generate_report(self, output_file: Path):
        """Generate JSON + TXT summaries in the run directory."""
        analysis = self.analyze()
        report = {
            "timestamp": datetime.now().isoformat(),
            "results_dir": str(self.results_dir),
            "analysis": analysis,
            "raw_metrics_count": len(self.metrics_data),
        }
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        summary_file = output_file.with_suffix(".txt")
        with summary_file.open("w", encoding="utf-8") as f:
            f.write("ENHANCED TERMINAL-BENCH ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 30 + "\n")
            s = analysis.get("summary", {})
            f.write(f"Total Tasks : {s.get('total_tasks', 0)}\n")
            f.write(f"Success Rate: {s.get('success_rate', 0.0):.2%}\n")
            f.write(f"Composite   : {s.get('avg_composite_score', 0.0):.3f}\n")
            f.write(f"Efficiency  : {s.get('avg_efficiency', 0.0):.3f}\n")
            f.write(f"Recovery    : {s.get('avg_recovery', 0.0):.3f}\n")
            f.write(f"Safety      : {s.get('avg_safety', 0.0):.3f}\n\n")

            f.write("FAILURE PATTERNS\n")
            f.write("-" * 30 + "\n")
            fp = analysis.get("failure_patterns", {})
            for err, cnt in fp.get("most_common_errors", []):
                f.write(f"  {err}: {cnt} occurrences\n")
            f.write("\n")

            f.write("LOOP ANALYSIS\n")
            f.write("-" * 30 + "\n")
            la = analysis.get("loop_analysis", {})
            f.write(f"Total Loops   : {la.get('total_loops_detected', 0)}\n")
            f.write(f"Tasks w/ Loops: {la.get('tasks_with_loops', 0)}\n")
            f.write(f"Loop Success  : {la.get('loop_success_rate', 0.0):.2%}\n\n")

            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(analysis.get("recommendations", []), 1):
                f.write(f"{i}. {rec}\n")

        return output_file, summary_file
