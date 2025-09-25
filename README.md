# Grok Terminal-Bench Evaluation

This repository evaluates Grok on the Terminal-Bench benchmark (from https://github.com/laude-institute/terminal-bench), focusing on AI agent performance in real terminal environments.

## Setup
1. Clone the repo: `git clone https://github.com/AndrewTKent/grok-benchmark-evaluation.git`
2. Run the setup script: `./setup.sh`
3. Edit `.env` with your `XAI_API_KEY` (from console.x.ai).
4. Activate venv: `source venv/bin/activate`

## Running the Benchmark
- Test connection: `python run.py --test`
- Run full benchmark: `python run.py --model grok-3 --dataset terminal-bench-core==0.1.1 --n-concurrent 4`
- Specific tasks: `python run.py --task-id task1 --task-id task2`

Results are saved to `results/` with timestamps.

## Replication
To replicate:
1. Ensure Docker is running.
2. Follow setup above.
3. Run the command from your evaluation (e.g., `python run.py ...`).
4. Compare with saved results in `results/`.

## Structure
- `src/`: Core Python code for Grok integration and running.
- `agents/`: Agent adapters for Terminal-Bench.

For more details, see the writeup document.