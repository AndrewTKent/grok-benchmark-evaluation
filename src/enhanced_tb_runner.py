# src/enhanced_tb_runner.py
"""Enhanced Terminal-Bench runner with composite scoring analysis"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import time
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tb_runner import TerminalBenchRunner, _Tailer


class EnhancedAnalyzer:
    """Analyzes enhanced metrics and generates insights"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.enhanced_metrics_dir = results_dir / "enhanced_metrics"
        self.metrics_data = []
        self.load_metrics()
    
    def load_metrics(self):
        """Load all enhanced metrics files"""
        if not self.enhanced_metrics_dir.exists():
            return
        
        for metrics_file in self.enhanced_metrics_dir.glob("metrics_*.json"):
            try:
                with open(metrics_file) as f:
                    self.metrics_data.append(json.load(f))
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive analysis"""
        if not self.metrics_data:
            return {"error": "No metrics data available"}
        
        analysis = {
            "summary": self._compute_summary(),
            "failure_patterns": self._analyze_failures(),
            "loop_analysis": self._analyze_loops(),
            "recovery_analysis": self._analyze_recovery(),
            "safety_analysis": self._analyze_safety(),
            "efficiency_analysis": self._analyze_efficiency(),
            "recommendations": self._generate_recommendations()
        }
        
        return analysis
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics"""
        df = pd.DataFrame(self.metrics_data)
        
        return {
            "total_tasks": len(df),
            "success_rate": df['success'].mean() if 'success' in df else 0,
            "avg_composite_score": df['composite_score'].mean() if 'composite_score' in df else 0,
            "avg_efficiency": df['efficiency_score'].mean() if 'efficiency_score' in df else 0,
            "avg_recovery": df['recovery_score'].mean() if 'recovery_score' in df else 0,
            "avg_safety": df['safety_score'].mean() if 'safety_score' in df else 0,
            "total_loops": df['loop_count'].sum() if 'loop_count' in df else 0,
            "avg_steps": df['steps_taken'].mean() if 'steps_taken' in df else 0,
            "avg_time": df['time_taken'].mean() if 'time_taken' in df else 0,
        }
    
    def _analyze_failures(self) -> Dict[str, Any]:
        """Analyze failure patterns"""
        error_counts = defaultdict(int)
        
        for metrics in self.metrics_data:
            if not metrics.get('success', False):
                for error_type, count in metrics.get('error_patterns', {}).items():
                    error_counts[error_type] += count
        
        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "most_common_errors": sorted_errors[:5],
            "total_error_types": len(error_counts),
            "error_distribution": dict(sorted_errors)
        }
    
    def _analyze_loops(self) -> Dict[str, Any]:
        """Analyze command loop patterns"""
        loop_tasks = []
        total_loops = 0
        
        for metrics in self.metrics_data:
            loops = metrics.get('loop_count', 0)
            if loops > 0:
                loop_tasks.append({
                    'task_id': metrics.get('task_id', 'unknown'),
                    'loops': loops,
                    'success': metrics.get('success', False)
                })
                total_loops += loops
        
        return {
            "total_loops_detected": total_loops,
            "tasks_with_loops": len(loop_tasks),
            "loop_success_rate": sum(1 for t in loop_tasks if t['success']) / len(loop_tasks) if loop_tasks else 0,
            "worst_offenders": sorted(loop_tasks, key=lambda x: x['loops'], reverse=True)[:5]
        }
    
    def _analyze_recovery(self) -> Dict[str, Any]:
        """Analyze recovery capabilities"""
        recovery_scores = []
        injected_failures = []
        
        for metrics in self.metrics_data:
            recovery_scores.append(metrics.get('recovery_score', 0))
            if 'injected_failures' in metrics:
                injected_failures.extend(metrics['injected_failures'])
        
        return {
            "avg_recovery_score": np.mean(recovery_scores) if recovery_scores else 0,
            "min_recovery_score": min(recovery_scores) if recovery_scores else 0,
            "max_recovery_score": max(recovery_scores) if recovery_scores else 0,
            "total_injected_failures": len(injected_failures),
            "recovery_std_dev": np.std(recovery_scores) if recovery_scores else 0
        }
    
    def _analyze_safety(self) -> Dict[str, Any]:
        """Analyze safety violations"""
        all_violations = []
        safety_scores = []
        
        for metrics in self.metrics_data:
            violations = metrics.get('safety_violations', [])
            all_violations.extend(violations)
            safety_scores.append(metrics.get('safety_score', 1.0))
        
        violation_patterns = defaultdict(int)
        for v in all_violations:
            if isinstance(v, dict):
                violation_patterns[v.get('pattern', 'unknown')] += 1
        
        return {
            "total_violations": len(all_violations),
            "avg_safety_score": np.mean(safety_scores) if safety_scores else 1.0,
            "violation_patterns": dict(violation_patterns),
            "high_risk_tasks": sum(1 for s in safety_scores if s < 0.5)
        }
    
    def _analyze_efficiency(self) -> Dict[str, Any]:
        """Analyze efficiency metrics"""
        df = pd.DataFrame(self.metrics_data)
        
        if df.empty:
            return {}
        
        efficiency_analysis = {
            "fastest_tasks": [],
            "slowest_tasks": [],
            "most_efficient": [],
            "least_efficient": []
        }
        
        if 'time_taken' in df.columns:
            sorted_by_time = df.sort_values('time_taken')
            efficiency_analysis['fastest_tasks'] = [
                {'task': row.get('task_id', 'unknown'), 'time': row['time_taken']}
                for _, row in sorted_by_time.head(3).iterrows()
            ]
            efficiency_analysis['slowest_tasks'] = [
                {'task': row.get('task_id', 'unknown'), 'time': row['time_taken']}
                for _, row in sorted_by_time.tail(3).iterrows()
            ]
        
        if 'efficiency_score' in df.columns:
            sorted_by_efficiency = df.sort_values('efficiency_score', ascending=False)
            efficiency_analysis['most_efficient'] = [
                {'task': row.get('task_id', 'unknown'), 'score': row['efficiency_score']}
                for _, row in sorted_by_efficiency.head(3).iterrows()
            ]
            efficiency_analysis['least_efficient'] = [
                {'task': row.get('task_id', 'unknown'), 'score': row['efficiency_score']}
                for _, row in sorted_by_efficiency.tail(3).iterrows()
            ]
        
        return efficiency_analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Analyze the data
        summary = self._compute_summary()
        failures = self._analyze_failures()
        loops = self._analyze_loops()
        
        # Success rate recommendations
        if summary['success_rate'] < 0.5:
            recommendations.append("Critical: Success rate below 50%. Focus on basic task completion.")
        
        # Loop detection recommendations
        if loops['total_loops_detected'] > len(self.metrics_data):
            recommendations.append("High loop frequency detected. Improve command variation strategies.")
        
        # Error pattern recommendations
        if failures['most_common_errors']:
            top_error = failures['most_common_errors'][0][0]
            if top_error == 'command_not_found':
                recommendations.append("Frequent 'command not found' errors. Add package installation logic.")
            elif top_error == 'permission_denied':
                recommendations.append("Permission issues common. Improve sudo/chmod handling.")
            elif top_error == 'file_not_found':
                recommendations.append("File path issues detected. Add existence checks before operations.")
        
        # Efficiency recommendations
        if summary['avg_efficiency'] < 0.5:
            recommendations.append("Low efficiency scores. Optimize command sequences and reduce steps.")
        
        # Safety recommendations
        if summary['avg_safety'] < 0.8:
            recommendations.append("Safety concerns detected. Review dangerous command patterns.")
        
        # Recovery recommendations
        if summary['avg_recovery'] < 0.6:
            recommendations.append("Poor error recovery. Implement better fallback strategies.")
        
        return recommendations
    
    def generate_report(self, output_file: Path):
        """Generate comprehensive analysis report"""
        analysis = self.analyze()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "results_dir": str(self.results_dir),
            "analysis": analysis,
            "raw_metrics_count": len(self.metrics_data)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also create a human-readable summary
        summary_file = output_file.with_suffix('.txt')
        with open(summary_file, 'w') as f:
            f.write("ENHANCED TERMINAL-BENCH ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary section
            f.write("SUMMARY\n")
            f.write("-" * 30 + "\n")
            summary = analysis['summary']
            f.write(f"Total Tasks: {summary['total_tasks']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.2%}\n")
            f.write(f"Composite Score: {summary['avg_composite_score']:.3f}\n")
            f.write(f"Efficiency: {summary['avg_efficiency']:.3f}\n")
            f.write(f"Recovery: {summary['avg_recovery']:.3f}\n")
            f.write(f"Safety: {summary['avg_safety']:.3f}\n\n")
            
            # Failure patterns
            f.write("FAILURE PATTERNS\n")
            f.write("-" * 30 + "\n")
            failures = analysis['failure_patterns']
            for error, count in failures['most_common_errors']:
                f.write(f"  {error}: {count} occurrences\n")
            f.write("\n")
            
            # Loop analysis
            f.write("LOOP ANALYSIS\n")
            f.write("-" * 30 + "\n")
            loops = analysis['loop_analysis']
            f.write(f"Total Loops: {loops['total_loops_detected']}\n")
            f.write(f"Tasks with Loops: {loops['tasks_with_loops']}\n")
            f.write(f"Loop Success Rate: {loops['loop_success_rate']:.2%}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(analysis['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        return output_file, summary_file


class EnhancedTerminalBenchRunner(TerminalBenchRunner):
    """Enhanced runner with composite scoring and analysis"""
    
    def __init__(self, 
                 model: str = None,
                 enable_enhanced_mode: bool = True,
                 enable_failure_injection: bool = False,
                 injection_rate: float = 0.1):
        super().__init__(model=model)
        self.enable_enhanced_mode = enable_enhanced_mode
        self.enable_failure_injection = enable_failure_injection
        self.injection_rate = injection_rate
        
        if self.enable_enhanced_mode:
            print("✓ Enhanced mode enabled with composite scoring")
            if self.enable_failure_injection:
                print(f"✓ Failure injection enabled (rate: {injection_rate})")
    
    def run_with_tb_cli(self,
                        dataset: str = "terminal-bench-core==0.1.1",
                        task_ids: Optional[List[str]] = None,
                        n_concurrent: int = 1,
                        n_attempts: int = 1,
                        timeout_per_task: int = 300) -> Dict[str, Any]:
        """
        Run Terminal-Bench with enhanced metrics if enabled.
        """
        # Verify setup
        if not self.verify_terminal_bench_setup():
            return {"status": "error", "message": "Setup verification failed"}
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_enhanced" if self.enable_enhanced_mode else ""
        output_dir = Path(f"results/tb_{self.model}{mode_suffix}_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = self._build_enhanced_command(
            dataset, task_ids, n_concurrent, n_attempts, output_dir
        )
        
        print(f"\nExecuting: {' '.join(cmd)}")
        print("=" * 60)
        
        # Set up environment
        env = self._prepare_enhanced_environment()
        
        # Start progress tailer
        tailer = _Tailer(output_dir)
        tailer.start()
        
        # Execute benchmark
        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line, end='')
                    output_lines.append(line)
            
            return_code = process.wait()
            elapsed_time = time.time() - start_time
            
            tailer.stop()
            
            # Parse results
            results = self._parse_enhanced_results(
                output_dir, return_code, output_lines, elapsed_time
            )
            
            # Run enhanced analysis if enabled
            if self.enable_enhanced_mode:
                analyzer = EnhancedAnalyzer(output_dir)
                analysis_file = output_dir / "enhanced_analysis.json"
                report_file, summary_file = analyzer.generate_report(analysis_file)
                
                results["enhanced_analysis"] = analyzer.analyze()
                results["analysis_files"] = {
                    "json": str(report_file),
                    "summary": str(summary_file)
                }
                
                print("\n" + "=" * 60)
                print("ENHANCED ANALYSIS")
                print("=" * 60)
                self._print_enhanced_summary(results["enhanced_analysis"])
            
            return results
            
        except KeyboardInterrupt:
            tailer.stop()
            print("\n✗ Interrupted by user")
            self._cleanup_docker_containers()
            return {"status": "interrupted"}
        except Exception as e:
            tailer.stop()
            print(f"\n✗ Error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _build_enhanced_command(self,
                                dataset: str,
                                task_ids: Optional[List[str]],
                                n_concurrent: int,
                                n_attempts: int,
                                output_dir: Path) -> List[str]:
        """Build command for enhanced or standard agent"""
        if self.enable_enhanced_mode:
            agent_path = "src.enhanced_agent:EnhancedGrokTerminalAgent"
        else:
            agent_path = "src.terminal_agent:GrokTerminalAgent"
        
        cmd = [
            "tb", "run",
            "--dataset", dataset,
            "--agent-import-path", agent_path,
            "--n-concurrent", str(n_concurrent),
            "--n-attempts", str(n_attempts),
            "--output-path", str(output_dir),
        ]
        
        # Add agent kwargs
        cmd.extend(["--agent-kwarg", f"model={self.model}"])
        
        if self.enable_enhanced_mode:
            cmd.extend(["--agent-kwarg", "enable_loop_detection=true"])
            
            if self.enable_failure_injection:
                cmd.extend(["--agent-kwarg", "enable_failure_injection=true"])
                cmd.extend(["--agent-kwarg", f"injection_rate={self.injection_rate}"])
        
        # Add task IDs
        if task_ids:
            for task_id in task_ids:
                cmd.extend(["--task-id", task_id])
        
        if self.debug:
            cmd.append("--verbose")
        
        return cmd
    
    def _prepare_enhanced_environment(self) -> Dict[str, str]:
        """Prepare environment with enhanced settings"""
        env = super()._prepare_environment()
        
        # Add enhanced mode flags
        if self.enable_enhanced_mode:
            env['ENHANCED_MODE'] = 'true'
            if self.enable_failure_injection:
                env['FAILURE_INJECTION'] = 'true'
        
        return env
    
    def _parse_enhanced_results(self,
                                output_dir: Path,
                                return_code: int,
                                output_lines: List[str],
                                elapsed_time: float) -> Dict[str, Any]:
        """Parse results including enhanced metrics"""
        results = super()._parse_results(output_dir, return_code, output_lines, elapsed_time)
        
        # Load enhanced metrics if available
        enhanced_dir = output_dir / "enhanced_metrics"
        if enhanced_dir.exists():
            metrics = []
            for metrics_file in enhanced_dir.glob("metrics_*.json"):
                try:
                    with open(metrics_file) as f:
                        metrics.append(json.load(f))
                except:
                    pass
            
            if metrics:
                results["enhanced_metrics"] = metrics
                results["enhanced_metrics_count"] = len(metrics)
        
        return results
    
    def _print_enhanced_summary(self, analysis: Dict[str, Any]):
        """Print enhanced analysis summary"""
        if "summary" in analysis:
            summary = analysis["summary"]
            print(f"\n📊 Composite Score: {summary['avg_composite_score']:.3f}")
            print(f"✅ Success Rate: {summary['success_rate']:.2%}")
            print(f"⚡ Efficiency: {summary['avg_efficiency']:.3f}")
            print(f"🔄 Recovery: {summary['avg_recovery']:.3f}")
            print(f"🛡️ Safety: {summary['avg_safety']:.3f}")
            
            if summary['total_loops'] > 0:
                print(f"⚠️  Loop Warnings: {summary['total_loops']} detected")
        
        if "recommendations" in analysis:
            print("\n📋 Recommendations:")
            for i, rec in enumerate(analysis["recommendations"][:3], 1):
                print(f"  {i}. {rec}")
    
    def run_comparative_analysis(self,
                                 dataset: str = "terminal-bench-core==0.1.1",
                                 task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run both standard and enhanced evaluation for comparison"""
        print("=" * 60)
        print("COMPARATIVE ANALYSIS MODE")
        print("=" * 60)
        print("Running benchmark twice: standard vs enhanced\n")
        
        # Run standard evaluation
        print("1. Running STANDARD evaluation...")
        self.enable_enhanced_mode = False
        standard_results = self.run_with_tb_cli(
            dataset=dataset,
            task_ids=task_ids,
            n_concurrent=1,
            n_attempts=1
        )
        
        # Run enhanced evaluation
        print("\n2. Running ENHANCED evaluation...")
        self.enable_enhanced_mode = True
        enhanced_results = self.run_with_tb_cli(
            dataset=dataset,
            task_ids=task_ids,
            n_concurrent=1,
            n_attempts=1
        )
        
        # Compare results
        comparison = {
            "standard": {
                "success_rate": standard_results.get("success_rate", 0),
                "output_dir": standard_results.get("output_dir")
            },
            "enhanced": {
                "composite_score": enhanced_results.get("enhanced_analysis", {})
                                    .get("summary", {}).get("avg_composite_score", 0),
                "success_rate": enhanced_results.get("success_rate", 0),
                "output_dir": enhanced_results.get("output_dir")
            },
            "improvement": {}
        }
        
        # Calculate improvements
        if comparison["standard"]["success_rate"] > 0:
            improvement = ((comparison["enhanced"]["success_rate"] - 
                          comparison["standard"]["success_rate"]) / 
                         comparison["standard"]["success_rate"] * 100)
            comparison["improvement"]["success_rate"] = f"{improvement:+.1f}%"
        
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"Standard Success Rate: {comparison['standard']['success_rate']:.2%}")
        print(f"Enhanced Success Rate: {comparison['enhanced']['success_rate']:.2%}")
        print(f"Enhanced Composite Score: {comparison['enhanced']['composite_score']:.3f}")
        
        if comparison["improvement"]:
            print(f"Improvement: {comparison['improvement']['success_rate']}")
        
        return comparison