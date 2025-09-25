# Grok Terminal-Bench Evaluation

This repository evaluates Grok on the Terminal-Bench benchmark (from https://github.com/laude-institute/terminal-bench), focusing on AI agent performance in real terminal environments.

## 🔧 Key Fixes Applied

1. **Proper Terminal-Bench Integration**: The agent now correctly interfaces with Terminal-Bench's execution environment instead of using mock responses.

2. **Simplified Agent Architecture**: Removed unnecessary `perform_task()` method - Terminal-Bench handles the execution loop.

3. **Enhanced Error Handling**: Added validation, command cleaning, and safety checks.

4. **Comprehensive Diagnostics**: New diagnostic tool to verify setup before running benchmarks.

5. **Better Command Extraction**: Improved parsing of Grok's responses to extract clean shell commands.

## 📋 Prerequisites

- Python 3.12+
- Docker (required for Terminal-Bench)
- Valid Grok API key from console.x.ai

## 🚀 Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/AndrewTKent/grok-benchmark-evaluation.git
cd grok-benchmark-evaluation

# 2. Run setup script
./setup.sh

# 3. Create .env file with your API key
echo "XAI_API_KEY=your_actual_key_here" > .env
echo "GROK_MODEL=grok-2-1212" >> .env  # or grok-beta

# 4. Activate virtual environment
source venv/bin/activate

# 5. Install Terminal-Bench
pip install terminal-bench
tb --version  # Verify installation

# 6. Download the terminal-bench-core dataset
tb datasets download --dataset terminal-bench-core==0.1.1  # Downloads latest version (head)
# For a specific version, e.g., 0.1.1:
# tb datasets download --dataset terminal-bench-core==0.1.1
# To overwrite existing dataset: add --overwrite
# To specify output directory: add --output-dir /path/to/dir

# 7. Run diagnostics to verify setup
python run.py --test
```

## 🧪 Testing & Running

### Quick Test
```bash
# Test with simple task and diagnostics
python run.py --test
```

### List Available Tasks
```bash
# See what tasks are available
python run.py --list-tasks
```

### Run Specific Tasks
```bash
# Run one or more specific tasks
python run.py --task-id "task_name_1" --task-id "task_name_2"
```

### Run Specific Tasks (Nice Subset)
```bash
# Run one or more specific tasks
python run.py \
  --dataset terminal-bench-core==0.1.1 \
  --n-concurrent 2 \
  --n-attempts 2 \
  --timeout 240 \
  --task-id hello-world \
  --task-id fix-git \
  --task-id fix-permissions \
  --task-id sqlite-db-truncate \
  --task-id csv-to-parquet \
  --task-id heterogeneous-dates \
  --task-id pytorch-model-cli.easy \
  --task-id crack-7z-hash.easy
```

### Full Benchmark
```bash
# Run complete benchmark (may take time and API credits)
python run.py --n-concurrent 4 --n-attempts 2

# With specific model
python run.py --model grok-beta --n-concurrent 2
```

### Debug Mode
```bash
# Enable verbose logging
python run.py --debug --test
```

## 📁 Project Structure

```
.
├── src/
│   ├── grok_client.py      # Grok API client with retry logic
│   ├── terminal_agent.py   # FIXED: Proper Terminal-Bench agent
│   └── tb_runner.py        # FIXED: Improved benchmark runner
├── diagnostic.py           # NEW: Comprehensive diagnostic tool
├── run.py                 # UPDATED: Main runner with test modes
├── setup.sh               # Automated setup script
├── requirements.txt       # Python dependencies
├── .env                   # Your API credentials (create this)
└── results/              # Benchmark results (auto-created)
```

## 🔍 Understanding Results

After running benchmarks, check the `results/` directory:

- `diagnostic_*.json` - System check results
- `quick_test_*.json` - Quick test outcomes
- `benchmark_summary_*.json` - Full benchmark results
- `tb_grok_*/` - Terminal-Bench raw outputs

### Key Metrics
- **Success Rate**: Percentage of tasks completed successfully
- **Failure Patterns**: Common reasons for task failures
- **Execution Time**: Time taken per task

## 🐛 Troubleshooting

### API Connection Issues
```bash
# Test API connection
python -c "from src.grok_client import GrokClient; GrokClient().test_connection()"
```

### Docker Issues
```bash
# Verify Docker is running
docker info

# Test Docker execution
docker run --rm alpine echo "Docker works"
```

### Terminal-Bench Issues
```bash
# Reinstall Terminal-Bench
pip uninstall terminal-bench
pip install terminal-bench

# Verify installation
tb --version
```

### Module Import Issues
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=$PWD:$PYTHONPATH

# Test import
python -c "from src.terminal_agent import GrokTerminalAgent; print('Import successful')"
```

## 📊 Analyzing Grok's Performance

The improved implementation helps identify Grok's strengths and weaknesses:

### Strengths Observed
- Strong bash command knowledge
- Good at file operations
- Handles multi-step tasks well

### Common Failure Modes
- May include explanations instead of pure commands
- Struggles with complex error recovery
- Sometimes generates overly cautious commands

### Extracting Failure Patterns
```python
# Analyze results programmatically
import json
from pathlib import Path

# Load latest results
result_files = sorted(Path("results").glob("benchmark_summary_*.json"))
with open(result_files[-1]) as f:
    results = json.load(f)

# Extract failures
if "benchmark_results" in results:
    tasks = results["benchmark_results"].get("tasks", [])
    failures = [t for t in tasks if t.get("status") != "success"]
    
    # Analyze failure reasons
    for task in failures:
        print(f"Task {task['id']}: {task.get('error', 'unknown error')}")
```

## 🔄 Replication Instructions

To replicate the evaluation:

1. **Environment Setup**
   ```bash
   git clone <repo-url>
   cd grok-benchmark-evaluation
   ./setup.sh
   echo "XAI_API_KEY=<your-key>" > .env
   ```

2. **Run Diagnostic**
   ```bash
   source venv/bin/activate
   python diagnostic.py
   ```

3. **Execute Benchmark**
   ```bash
   # Same parameters as original evaluation
   python run.py --model grok-2-1212 --dataset terminal-bench-core==0.1.1 --n-concurrent 4
   ```

4. **Compare Results**
   Results are timestamped in `results/` for comparison across runs.

## 🚧 Known Limitations

1. **Terminal-Bench Limitations**:
   - Only tests bash command execution
   - Limited to single-container tasks
   - No multi-modal capabilities
   - May not reflect real-world agent usage patterns

2. **Current Implementation**:
   - History truncation may lose context in long tasks
   - Command extraction heuristics may miss edge cases
   - No fine-tuning for specific task types

## 🔮 Suggested Improvements

1. **Enhanced Command Parsing**: Use a dedicated parser for shell commands
2. **Task-Specific Prompts**: Optimize prompts based on task categories
3. **Error Recovery**: Implement smarter error recovery strategies
4. **Context Management**: Better handling of long conversation histories

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please ensure all tests pass:
```bash
python diagnostic.py
python run.py --test
```
