import json, time, subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from src.runners.tb_runner_base import TBRunnersBase, _Tailer, _sanitize_for_fs

class StandardTerminalBenchRunner(TBRunnersBase):
    def test_connection(self) -> bool:
        print("Testing Grok API connection...")
        try:
            resp = self.client.generate("Respond with 'OK' and nothing else.", temperature=0.0, max_tokens=10)
            ok = isinstance(resp, str) and "OK" in resp
            print("✓ Connection successful" if ok else f"✗ Unexpected: {resp}")
            return ok
        except Exception as e:
            print(f"✗ Connection failed: {e}"); return False

    def run_diagnostic_test(self) -> Dict[str, Any]:
        from src.agents.grok_terminal_agent import GrokTerminalAgent
        print("\n====== RUNNING DIAGNOSTIC TEST ======")
        results = {"timestamp": datetime.now().isoformat(), "model": self.model, "checks": {}}
        results["checks"]["api"] = self.test_connection()
        results["checks"]["setup"] = self.verify_terminal_bench_setup()
        try:
            agent = GrokTerminalAgent(model=self.model)
            action = agent.get_action("$ pwd\n/home/user\n$ ")
            print(f"✓ Agent instantiated. sample action: {action}")
            results["checks"]["agent"] = True
        except Exception as e:
            print(f"✗ Agent failed: {e}"); results["checks"]["agent"] = False
        # simple docker test
        try:
            r = subprocess.run("docker run --rm alpine echo 'Docker OK'", shell=True, capture_output=True, text=True, timeout=10)
            results["checks"]["docker"] = "Docker OK" in r.stdout
            print("✓ Docker exec working" if results["checks"]["docker"] else "✗ Docker exec failed")
        except Exception as e:
            print(f"✗ Docker test error: {e}"); results["checks"]["docker"] = False
        # save
        out = Path("results/diagnostic_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved diagnostic to: {out}")
        return results

    def run_quick_test(self) -> int:
        # same semantics as your previous run.py quick test, minimized
        ok = self.run_diagnostic_test()
        if not all(ok["checks"].values()): return 1
        print("\n====== RUNNING SIMPLE TB TEST ======")
        res = self.run_with_tb_cli(dataset="terminal-bench-core==0.1.1", task_ids=None, n_concurrent=1, n_attempts=1, timeout_per_task=60)
        out = Path("results/quick_test_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
        out.write_text(json.dumps(res, indent=2))
        print(f"Saved test results to: {out}")
        return 0

    def run_with_tb_cli(self, dataset: str, task_ids: Optional[List[str]], n_concurrent: int, n_attempts: int, timeout_per_task: int) -> Dict[str, Any]:
        if not self.verify_terminal_bench_setup():
            return {"status":"error","message":"setup verification failed"}
        output_dir = Path(f"results/tb_{_sanitize_for_fs(self.model)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["tb","run","--dataset",dataset,"--agent-import-path","src.agents.grok_terminal_agent:GrokTerminalAgent",
               "--n-concurrent",str(n_concurrent),"--n-attempts",str(n_attempts),"--output-path",str(output_dir),
               "--agent-kwarg",f"model={self.model}"]
        if task_ids:
            for t in task_ids: cmd += ["--task-id", t]
        if self.debug: cmd.append("--verbose")
        print(f"\nExecuting: {' '.join(cmd)}\n" + "="*60)
        env = self._prepare_environment()
        tailer = _Tailer(output_dir); tailer.start()
        start = time.time(); output_lines=[]
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(proc.stdout.readline, ''):
                if line: print(line, end=''); output_lines.append(line)
            rc = proc.wait(); elapsed = time.time()-start; tailer.stop()
            print(f"\n{'='*60}\nExecution completed in {elapsed:.1f}s")
            return self._parse_results(output_dir, rc, output_lines, elapsed)
        except KeyboardInterrupt:
            tailer.stop(); print("\n✗ Interrupted"); return {"status":"interrupted"}
        except Exception as e:
            tailer.stop(); print(f"\n✗ Error: {e}"); return {"status":"error","message":str(e)}
