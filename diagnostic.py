#!/usr/bin/env python3
"""Quick diagnostic for Terminal-Bench + Grok setup"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.tb_runner import TerminalBenchRunner

runner = TerminalBenchRunner()
results = runner.run_diagnostic_test()
sys.exit(0 if all(results["checks"].values()) else 1)
