# src/runners/enhanced_runner.py
"""Enhanced Terminal-Bench runner with composite scoring analysis (refactored)."""

import json
import time
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.runners.tb_runner_base import TBRunnersBase, _Tailer, _sanitize_for_fs
from src.analysis.analyzer import EnhancedAnalyzer


class EnhancedTerminalBenchRunner(TBRunnersBase):
    """Enhanced runner that can switch between standard/enhanced agents and run post-hoc analysis."""

    def __init__(
        self,
        model: str = None,
        enable_enhanced_mode: bool = True,
        enable_failure_injection: bool = False,
        injection_rate: float = 0.1,
    ):
        super().__init__(model=model)
        self.enable_enhanced_mode = enable_enhanced_mode
        self.enable_failure_injection = enable_failure_injection
        self.injection_rate = injection_rate

        if self.enable_enhanced_mode:
            print("✓ Enhanced mode enabled")
            if self.enable_failure_injection:
                print(f"✓ Failure injection enabled (rate={self.injection_rate})")

    # ----- public API -----

    def run_with_tb_cli(
        self,
        dataset: str = "terminal-bench-core==0.1.1",
        task_ids: Optional[List[str]] = None,
        n_concurrent: int = 1,
        n_attempts: int = 1,
        timeout_per_task: int = 300,
    ) -> Dict[str, Any]:
        """Run Terminal-Bench with enhanced metrics if enabled (otherwise standard agent)."""
        if not self.verify_terminal_bench_setup():
            return {"status": "error", "message": "Setup verification failed"}

        # Output directory (distinct name if enhanced)
        sanitized_model = _sanitize_for_fs(self.model)
        suffix = "_enhanced" if self.enable_enhanced_mode else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/tb_{sanitized_model}{suffix}_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build and execute
        cmd = self._build_command(dataset, task_ids, n_concurrent, n_attempts, output_dir)
        print(f"\nExecuting: {' '.join(cmd)}")
        print("=" * 60)

        env = self._prepare_environment()
        tailer = _Tailer(output_dir)
        tailer.start()

        start_time = time.time()
        output_lines: List[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(proc.stdout.readline, ''):
                if line:
                    print(line, end='')
                    output_lines.append(line)

            return_code = proc.wait()
            elapsed_time = time.time() - start_time
            tailer.stop()

            print(f"\n{'=' * 60}\nExecution completed in {elapsed_time:.1f} seconds")

            # Parse results (baseline TB artifacts)
            results = self._parse_results(output_dir, return_code, output_lines, elapsed_time)

            # If enhanced mode, run analysis over enhanced metrics if present
            if self.enable_enhanced_mode:
                analyzer = EnhancedAnalyzer(output_dir)
                analysis_file = output_dir / "enhanced_analysis.json"
                report_json, report_txt = analyzer.generate_report(analysis_file)
                results["enhanced_analysis"] = analyzer.analyze()
                results["analysis_files"] = {"json": str(report_json), "summary": str(report_txt)}

                self._print_enhanced_summary(results.get("enhanced_analysis", {}))

            return results

        except KeyboardInterrupt:
            tailer.stop()
            print("\n✗ Interrupted by user")
            return {"status": "interrupted"}
        except Exception as e:
            tailer.stop()
            print(f"\n✗ Error: {e}")
            return {"status": "error", "message": str(e)}

    def run_comparative_analysis(
        self,
        dataset: str = "terminal-bench-core==0.1.1",
        task_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run both standard and enhanced evaluations for comparison."""
        print("=" * 60)
        print("COMPARATIVE ANALYSIS MODE")
        print("=" * 60)

        # 1) Standard
        print("\n1) Running STANDARD evaluation...")
        original_flags = (self.enable_enhanced_mode, self.enable_failure_injection, self.injection_rate)
        self.enable_enhanced_mode = False
        standard = self.run_with_tb_cli(
            dataset=dataset, task_ids=task_ids, n_concurrent=1, n_attempts=1
        )

        # 2) Enhanced (without injection by default)
        print("\n2) Running ENHANCED evaluation...")
        self.enable_enhanced_mode, self.enable_failure_injection, self.injection_rate = True, False, original_flags[2]
        enhanced = self.run_with_tb_cli(
            dataset=dataset, task_ids=task_ids, n_concurrent=1, n_attempts=1
        )

        # Restore flags
        self.enable_enhanced_mode, self.enable_failure_injection, self.injection_rate = original_flags

        comparison = {
            "standard": {
                "success_rate": standard.get("success_rate", 0.0),
                "output_dir": standard.get("output_dir"),
            },
            "enhanced": {
                "success_rate": enhanced.get("success_rate", 0.0),
                "composite_score": (enhanced.get("enhanced_analysis", {})
                                    .get("summary", {})
                                    .get("avg_composite_score", 0.0)),
                "output_dir": enhanced.get("output_dir"),
            },
            "improvement": {},
        }
        s = comparison["standard"]["success_rate"]
        e = comparison["enhanced"]["success_rate"]
        if s:
            comparison["improvement"]["success_rate_pct"] = f"{(e - s) / s * 100:+.1f}%"

        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"Standard Success Rate : {s:.2%}")
        print(f"Enhanced Success Rate : {e:.2%}")
        print(f"Enhanced Composite    : {comparison['enhanced']['composite_score']:.3f}")
        if "success_rate_pct" in comparison["improvement"]:
            print(f"Success Δ            : {comparison['improvement']['success_rate_pct']}")

        # Persist comparison summary
        out = Path("results/comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(comparison, indent=2))
        print(f"\nSaved comparison to: {out}")

        return comparison

    # ----- internals -----

    def _build_command(
        self,
        dataset: str,
        task_ids: Optional[List[str]],
        n_concurrent: int,
        n_attempts: int,
        output_dir: Path,
    ) -> List[str]:
        """Construct the tb CLI command for standard or enhanced agent."""
        if self.enable_enhanced_mode:
            agent_path = "src.agents.enhanced_grok_agent:EnhancedGrokTerminalAgent"
        else:
            agent_path = "src.agents.grok_terminal_agent:GrokTerminalAgent"

        cmd = [
            "tb", "run",
            "--dataset", dataset,
            "--agent-import-path", agent_path,
            "--n-concurrent", str(n_concurrent),
            "--n-attempts", str(n_attempts),
            "--output-path", str(output_dir),
            "--agent-kwarg", f"model={self.model}",
        ]

        if self.enable_enhanced_mode and self.enable_failure_injection:
            cmd.extend(["--agent-kwarg", "enable_failure_injection=true"])
            cmd.extend(["--agent-kwarg", f"injection_rate={self.injection_rate}"])
        elif self.enable_enhanced_mode:
            cmd.extend(["--agent-kwarg", "enable_loop_detection=true"])

        if task_ids:
            for t in task_ids:
                cmd.extend(["--task-id", t])

        if self.debug:
            cmd.append("--verbose")

        return cmd

    def _print_enhanced_summary(self, analysis: Dict[str, Any]) -> None:
        """Pretty-print a short summary of enhanced analysis results."""
        if not analysis:
            print("\n(No enhanced analysis available)")
            return

        print("\n" + "=" * 60)
        print("ENHANCED ANALYSIS SUMMARY")
        print("=" * 60)

        summary = analysis.get("summary", {})
        if summary:
            print(f"📊 Composite Score : {summary.get('avg_composite_score', 0.0):.3f}")
            print(f"✅ Success Rate    : {summary.get('success_rate', 0.0):.2%}")
            print(f"⚡ Efficiency      : {summary.get('avg_efficiency', 0.0):.3f}")
            print(f"🔄 Recovery        : {summary.get('avg_recovery', 0.0):.3f}")
            print(f"🛡️  Safety          : {summary.get('avg_safety', 0.0):.3f}")
            print(f"⚠️  Loop Warnings   : {summary.get('total_loops', 0)}")

        recs = analysis.get("recommendations", [])
        if recs:
            print("\n📋 Top Recommendations:")
            for i, rec in enumerate(recs[:3], 1):
                print(f"  {i}. {rec}")
