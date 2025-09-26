import argparse
from pathlib import Path
from src.runners.standard_runner import StandardTerminalBenchRunner
from src.runners.enhanced_runner import EnhancedTerminalBenchRunner

def main() -> int:
    p = argparse.ArgumentParser(
        description="Run Terminal-Bench with Grok (standard & enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # common flags
    p.add_argument("--model", help="Model name (overrides .env)")
    p.add_argument("--dataset", default="terminal-bench-core==0.1.1")
    p.add_argument("--task-id", action="append", dest="task_ids")
    p.add_argument("--n-concurrent", type=int, default=5)
    p.add_argument("--n-attempts", type=int, default=1)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--debug", action="store_true")

    # enhanced toggles (no behavior change yet if not used)
    p.add_argument("--enhanced", action="store_true", help="Use enhanced agent")
    p.add_argument("--inject-failures", action="store_true")
    p.add_argument("--injection-rate", type=float, default=0.1)

    # comparison harness (will call both paths)
    p.add_argument("--compare", action="store_true", help="Run standard vs enhanced comparison")

    args = p.parse_args()

    # choose runner
    if args.enhanced or args.compare:
        runner = EnhancedTerminalBenchRunner(
            model=args.model,
            enable_enhanced_mode=True,
            enable_failure_injection=args.inject_failures,
            injection_rate=args.injection_rate,
        )
    else:
        runner = StandardTerminalBenchRunner(model=args.model)

    if args.debug:
        import os
        os.environ["GROK_DEBUG"] = "true"

    # modes
    if args.diagnostic:
        ok = runner.run_diagnostic_test()
        return 0 if all(ok["checks"].values()) else 1

    if args.list_tasks:
        return runner.list_available_tasks(dataset=args.dataset)

    if args.test:
        return runner.run_quick_test()

    if args.compare:
        # one-call comparison (standard vs enhanced)
        comp = runner.run_comparative_analysis(dataset=args.dataset, task_ids=args.task_ids)
        print("\nComparison summary:", comp)
        return 0

    # default: run once (standard or enhanced depending on flag)
    results = runner.run_with_tb_cli(
        dataset=args.dataset,
        task_ids=args.task_ids,
        n_concurrent=args.n_concurrent,
        n_attempts=args.n_attempts,
        timeout_per_task=args.timeout,
    )
    return 0 if results.get("status") in {"completed"} else 1
