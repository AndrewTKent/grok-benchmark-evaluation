import os, re, json, time, glob, shutil, logging, subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

def _shorten(s: str, n: int = 120) -> str:
    s = s or "";  return s if len(s) <= n else s[: n - 1] + "…"

def _sanitize_for_fs(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")

class _Tailer:
    def __init__(self, output_dir: Path, poll_interval: float = 1.0):
        self.output_dir = output_dir
        self.poll_interval = poll_interval
        self._stop = False
        self._jsonl_pos: Dict[Path, int] = {}
        self._cmd_pos: Dict[Path, int] = {}

    def start(self): import threading; threading.Thread(target=self._loop, daemon=True).start()
    def stop(self): self._stop = True

    def _loop(self):
        print("\n== Live Task Progress (progress.jsonl / commands.txt) ==")
        while not self._stop:
            try:
                jsonl_paths = glob.glob(str(self.output_dir / "*" / "*" / "*" / "agent-logs" / "progress.jsonl"))
                cmd_paths   = glob.glob(str(self.output_dir / "*" / "*" / "*" / "commands.txt"))
                for p in jsonl_paths:
                    path = Path(p)
                    pos = self._jsonl_pos.get(path, 0)
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            f.seek(pos)
                            for line in f:
                                line = line.strip()
                                if not line: continue
                                try: evt = json.loads(line)
                                except json.JSONDecodeError: continue
                                task = self._extract_task_from_path(path)
                                phase = evt.get("phase", "?"); msg = evt.get("msg","")
                                step, total = evt.get("step"), evt.get("total_steps")
                                step_part = f"[{step}/{total}]" if step and total else ""
                                extra = evt.get("extra") or {}
                                preview = f"  cmd: {_shorten(extra['cmd_preview'],100)}" if "cmd_preview" in extra else ""
                                print(f"[TB_PROGRESS] {task} {step_part} {phase}: {msg}{(' | '+preview) if preview else ''}")
                            self._jsonl_pos[path] = f.tell()
                    except (FileNotFoundError, PermissionError):
                        continue
                for p in cmd_paths:
                    path = Path(p)
                    pos = self._cmd_pos.get(path, 0)
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            f.seek(pos)
                            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                            if lines:
                                task = self._extract_task_from_path(path)
                                print(f"[TB_CMD] {task} -> {_shorten(lines[-1], 140)}")
                            self._cmd_pos[path] = f.tell()
                    except (FileNotFoundError, PermissionError):
                        continue
                time.sleep(self.poll_interval)
            except Exception:
                time.sleep(self.poll_interval)

    @staticmethod
    def _extract_task_from_path(path: Path) -> str:
        parts = path.parts
        try:
            trial = parts[-3]; task = parts[-4]
            return f"{task}/{trial}"
        except Exception:
            return str(path.parent)

class TBRunnersBase:
    def __init__(self, model: Optional[str] = None):
        from src.clients.grok_client import GrokClient
        self.model = model or os.getenv('GROK_MODEL') or os.getenv('XAI_MODEL') or 'grok-4-fast-reasoning'
        self.client = GrokClient(model=self.model)
        self.debug = os.getenv('GROK_DEBUG', 'false').lower() == 'true'

    # shared setup/verify
    def verify_terminal_bench_setup(self) -> bool:
        print("\nVerifying Terminal-Bench setup...")
        ok = True; log = logging.getLogger(__name__)
        if shutil.which('tb') is None:
            print("✗ tb CLI not found. Try: pip install terminal-bench"); ok = False
        else:
            try:
                r = subprocess.run(["tb","--help"], capture_output=True, text=True, timeout=20)
                if r.returncode == 0: print("✓ Terminal-Bench CLI is operational")
                else: print(f"✗ tb failed: {r.stderr.strip()}"); ok = False
            except Exception as e: print(f"✗ tb check error: {e}"); ok = False
        try:
            r = subprocess.run(["docker","info"], capture_output=True, text=True, timeout=8)
            if r.returncode == 0: print("✓ Docker is running")
            else: print("✗ Docker not running"); ok = False
        except Exception as e: print(f"✗ Docker check error: {e}"); ok = False
        try:
            from src.agents.grok_terminal_agent import GrokTerminalAgent  # noqa
            print("✓ GrokTerminalAgent importable")
        except Exception as e:
            print(f"✗ GrokTerminalAgent import failed: {e}"); ok = False
        return ok

    def _prepare_environment(self) -> Dict[str,str]:
        env = os.environ.copy()
        cwd = str(Path.cwd())
        env['PYTHONPATH'] = f"{cwd}:{env.get('PYTHONPATH','')}".rstrip(":")
        for var in ['XAI_API_KEY','GROK_API_KEY','GROK_MODEL','GROK_DEBUG','XAI_MODEL']:
            if var in os.environ: env[var] = os.environ[var]
        return env

    def _parse_results(self, output_dir: Path, return_code: int, output_lines: List[str], elapsed_time: float) -> Dict[str, Any]:
        results: Dict[str,Any] = {
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "elapsed_time": elapsed_time,
            "output_dir": str(output_dir)
        }
        for pattern in ["results.json","summary.json","*.json"]:
            for f in output_dir.glob(pattern):
                try:
                    with f.open(encoding="utf-8") as fh:
                        data = json.load(fh)
                        results["benchmark_results"] = data
                        if isinstance(data, dict) and "tasks" in data:
                            tasks = data["tasks"]; successes = sum(1 for t in tasks if t.get("status")=="success")
                            results["num_tasks"] = len(tasks)
                            if tasks: results["success_rate"] = successes/len(tasks)
                        break
                except Exception:
                    continue
        if "benchmark_results" not in results: results["raw_output"] = output_lines
        return results

    # util for listing tasks (extracted from your run.py)
    def list_available_tasks(self, dataset: str) -> int:
        import re, subprocess
        name, ver = dataset.split("==", 1)
        try:
            dl = subprocess.run(["tb","datasets","download","--dataset",dataset], capture_output=True, text=True, timeout=120)
        except FileNotFoundError:
            print("❌ tb not found. pip install terminal-bench"); return 1
        m = re.search(r"Dataset location:\s*(.+)", dl.stdout) or re.search(r"Dataset location:\s*(.+)", dl.stderr)
        from pathlib import Path as P
        dataset_path = P(m.group(1).strip()) if m else (P.home()/".cache"/"terminal-bench"/name/ver)
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {dataset_path}"); return 1
        entries = sorted(p.name for p in dataset_path.iterdir() if p.is_dir())
        tasks = [n for n in entries if (dataset_path/n/"task.yaml").exists()] or entries
        print(f"📁 {name}@{ver}\n📂 {dataset_path}\n📝 {len(tasks)} tasks:\n")
        for t in tasks: print(f"- {t}")
        return 0
