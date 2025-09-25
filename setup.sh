#!/bin/bash

#####################################################################
#  Enhanced Setup Script for Grok Terminal-Bench Evaluation        #
#  Version 2.1 - Prefer Python 3.12 + venv-first execution         #
#####################################################################

clear
set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
VENV_DIR="venv"
TBENCH_REPO="https://github.com/laude-institute/terminal-bench.git"
TBENCH_DIR="terminal-bench"
PYTHON_MIN_VERSION="3.12"

# Helper functions
print_header() {
    echo -e "\n${BLUE}${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}${BOLD}  $1${NC}"
    echo -e "${BLUE}${BOLD}═══════════════════════════════════════════════════════${NC}\n"
}

print_status()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error()   { echo -e "${RED}[✗]${NC} $1"; }

check_command() {
    command -v "$1" >/dev/null 2>&1
}

# Parse arguments
FORCE_REINSTALL=false
SKIP_TESTS=false
CLONE_TBENCH=true
FIX_TB_CLI=true  # New flag for fixing tb CLI issues

while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)         FORCE_REINSTALL=true; shift ;;
        --skip-tests)       SKIP_TESTS=true; shift ;;
        --no-clone-tbench)  CLONE_TBENCH=false; shift ;;
        --no-fix-tb)        FIX_TB_CLI=false; shift ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo "Options:"
            echo "  -f, --force         Force reinstall all components"
            echo "  --skip-tests        Skip connection tests"
            echo "  --no-clone-tbench   Skip cloning terminal-bench repository"
            echo "  --no-fix-tb         Skip Terminal-Bench CLI fixes"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *) print_error "Unknown option: $1"; exit 1 ;;
    esac
done

print_header "Terminal-Bench (T-Bench) Environment Setup v2.1"

# ASCII art for T-Bench
echo -e "${BLUE}"
cat << 'EOF'
  _____                   _             _     ____                  _     
 |_   _|__ _ __ _ __ ___ (_)_ __   __ _| |   | __ )  ___ _ __   ___| |__  
   | |/ _ \ '__| '_ ` _ \| | '_ \ / _` | |   |  _ \ / _ \ '_ \ / __| '_ \ 
   | |  __/ |  | | | | | | | | | | (_| | |   | |_) |  __/ | | | (__| | | |
   |_|\___|_|  |_| |_| |_|_|_| |_|\__,_|_|   |____/ \___|_| |_|\___|_| |_|
                                                             + Grok Integration
EOF
echo -e "${NC}"

# 1. Check system dependencies
print_header "Checking System Dependencies"

# Decide Python interpreter: prefer python3.12
PY=""
if check_command python3.12; then
  PY="python3.12"
elif check_command python3; then
  PY="python3"
else
  print_error "No python3 found. Please install Python ${PYTHON_MIN_VERSION}+."
  exit 1
fi

# Verify chosen interpreter meets minimum version
ver="$($PY -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
req="${PYTHON_MIN_VERSION}"
if [ "$(printf '%s\n' "$req" "$ver" | sort -V | head -n1)" != "$req" ]; then
  print_error "Found $($PY --version 2>&1). Python ${PYTHON_MIN_VERSION}+ is required."
  print_error "Install python3.12 (e.g., deadsnakes on Ubuntu) and rerun."
  exit 1
fi
print_success "Using Python interpreter: $($PY --version 2>&1)"

# Check git
if check_command git; then
    print_success "Git is installed"
else
    print_error "Git is not installed. Please install git first."
    exit 1
fi

# Check Docker (REQUIRED for terminal-bench)
if check_command docker; then
    print_success "Docker is installed"
    if docker info >/dev/null 2>&1; then
        print_success "Docker daemon is running"
    else
        print_error "Docker is installed but not running. Please start Docker."
        exit 1
    fi
else
    print_error "Docker is REQUIRED for Terminal-Bench. Please install and start Docker."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# 2. Setup Python virtual environment
print_header "Python Virtual Environment"

if [ -d "$VENV_DIR" ] && [ "$FORCE_REINSTALL" = false ]; then
    print_status "Virtual environment exists, activating..."
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    if [ -n "$VIRTUAL_ENV" ]; then
        print_success "Virtual environment activated"
    else
        print_warning "Failed to activate. Recreating..."
        rm -rf "$VENV_DIR"
        "$PY" -m venv "$VENV_DIR"
        # shellcheck disable=SC1090
        source "$VENV_DIR/bin/activate"
        print_success "Virtual environment recreated"
    fi
else
    if [ "$FORCE_REINSTALL" = true ] && [ -d "$VENV_DIR" ]; then
        print_status "Force reinstall: Removing existing venv..."
        rm -rf "$VENV_DIR"
    fi
    print_status "Creating virtual environment with $PY..."
    "$PY" -m venv "$VENV_DIR"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment created and activated"
fi

# Define venv executables
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# Upgrade pip (venv)
print_status "Upgrading pip..."
"$VENV_PY" -m pip install --quiet --upgrade pip

# 3. Clone/Update Terminal-Bench repository
if [ "$CLONE_TBENCH" = true ]; then
    print_header "Terminal-Bench Repository"
    if [ -d "$TBENCH_DIR" ]; then
        if [ "$FORCE_REINSTALL" = true ]; then
            print_status "Force reinstall: Re-cloning terminal-bench..."
            rm -rf "$TBENCH_DIR"
            git clone "$TBENCH_REPO" "$TBENCH_DIR"
            print_success "terminal-bench cloned fresh"
        else
            print_status "terminal-bench exists. Checking for updates..."
            pushd "$TBENCH_DIR" >/dev/null
            if [ -d ".git" ]; then
                git fetch origin -q || true
                LOCAL=$(git rev-parse @ 2>/dev/null || echo "")
                REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "")
                if [ -n "$LOCAL" ] && [ -n "$REMOTE" ] && [ "$LOCAL" != "$REMOTE" ]; then
                    print_warning "Updates available for terminal-bench"
                    read -p "Update to latest version? (y/n) " -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        git pull --ff-only origin main || git pull origin main
                        print_success "Updated to latest version"
                    fi
                else
                    current_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
                    print_success "terminal-bench is up to date (commit: $current_commit)"
                fi
            fi
            popd >/dev/null
        fi
    else
        print_status "Cloning terminal-bench repository..."
        git clone "$TBENCH_REPO" "$TBENCH_DIR"
        print_success "terminal-bench cloned"
    fi
fi

# 4. Install Terminal-Bench with fixes for CLI issues
print_header "Installing Terminal-Bench"

if [ "$FORCE_REINSTALL" = true ]; then
    print_status "Uninstalling existing terminal-bench (if any)..."
    "$VENV_PIP" uninstall -y terminal-bench 2>/dev/null || true
    "$VENV_PIP" cache purge 2>/dev/null || true
fi

print_status "Installing terminal-bench package..."
if [ -d "$TBENCH_DIR" ] && { [ -f "$TBENCH_DIR/setup.py" ] || [ -f "$TBENCH_DIR/pyproject.toml" ]; }; then
    "$VENV_PIP" install -e "$TBENCH_DIR"/
    print_success "terminal-bench installed from local repository"
else
    "$VENV_PIP" install --upgrade terminal-bench
    print_success "terminal-bench package installed from PyPI"
fi

# 5. Fix Terminal-Bench CLI timeout issues
if [ "$FIX_TB_CLI" = true ]; then
    print_header "Fixing Terminal-Bench CLI"
    print_status "Creating Terminal-Bench wrapper scripts..."

    # tb-safe: Python wrapper that handles quick commands with a short timeout
    cat > "$VENV_DIR/bin/tb-safe" << 'EOF'
#!/usr/bin/env python3
"""Safe wrapper for Terminal-Bench CLI that handles timeouts gracefully."""
import sys, subprocess

def run(args, timeout=10):
    try:
        if '--version' in args or '--help' in args:
            r = subprocess.run(['python', '-m', 'terminal_bench'] + args[1:],
                               capture_output=True, text=True, timeout=timeout)
            if r.stdout: print(r.stdout, end="")
            if r.stderr: print(r.stderr, file=sys.stderr, end="")
            return r.returncode
        else:
            return subprocess.run(['python', '-m', 'terminal_bench'] + args[1:]).returncode
    except subprocess.TimeoutExpired:
        print("Terminal-Bench is initializing (first-run dataset/setup). Try again, or use 'python -m terminal_bench'.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user"); return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); return 1

if __name__ == "__main__":
    sys.exit(run(sys.argv))
EOF
    chmod +x "$VENV_DIR/bin/tb-safe"
    print_success "Created tb-safe wrapper"

    # tb-direct: direct module runner
    cat > "$VENV_DIR/bin/tb-direct" << 'EOF'
#!/bin/bash
# Direct Terminal-Bench runner using Python module
exec python -m terminal_bench "$@"
EOF
    chmod +x "$VENV_DIR/bin/tb-direct"
    print_success "Created tb-direct wrapper"

    # Test Terminal-Bench import using venv Python
    print_status "Testing Terminal-Bench Python module..."
    if "$VENV_PY" -c "import terminal_bench; print('✓ Module imports successfully')" >/dev/null 2>&1; then
        print_success "Terminal-Bench Python module works"
    else
        print_error "Terminal-Bench module import failed"
    fi

    # Try to initialize Terminal-Bench (non-fatal)
    print_status "Initializing Terminal-Bench (may download datasets on first run)..."
    if command -v timeout >/dev/null 2>&1; then
        timeout 30 "$VENV_PY" - << 'PYINIT' 2>/dev/null || true
try:
    import terminal_bench
    print('✓ Terminal-Bench initialized')
except Exception as e:
    print(f'⚠ Partial initialization: {e}')
PYINIT
    else
        "$VENV_PY" - << 'PYINIT' 2>/dev/null || true
try:
    import terminal_bench
    print('✓ Terminal-Bench initialized')
except Exception as e:
    print(f'⚠ Partial initialization: {e}')
PYINIT
    fi
fi

# 6. Install additional requirements
print_header "Installing Dependencies"

if [ -f "requirements.txt" ]; then
    print_status "Installing requirements.txt..."
    "$VENV_PIP" install -q -r requirements.txt
    print_success "Requirements installed"
else
    print_status "Installing common dependencies..."
    "$VENV_PIP" install requests python-dotenv pandas tqdm jsonlines numpy openai aiohttp pyyaml
    print_success "Common dependencies installed"
fi

# 7. Create .env template if it doesn't exist
if [ ! -f ".env" ]; then
    print_header "Creating Environment Configuration"
    cat > .env << 'EOF'
# Grok API Configuration
XAI_API_KEY=your_actual_key_here
GROK_MODEL=grok-2-1212

# Optional settings
GROK_DEBUG=false
GROK_TIMEOUT=60
GROK_MAX_RETRIES=3
EOF
    print_success "Created .env template"
    print_warning "Please edit .env and add your XAI_API_KEY from console.x.ai"
else
    print_success ".env file exists"
fi

# 8. Create helper scripts
print_header "Creating Helper Scripts"

# diagnostic.py
cat > diagnostic.py << 'EOF'
#!/usr/bin/env python3
"""Quick diagnostic for Terminal-Bench + Grok setup"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.tb_runner import TerminalBenchRunner

runner = TerminalBenchRunner()
results = runner.run_diagnostic_test()
sys.exit(0 if all(results["checks"].values()) else 1)
EOF
chmod +x diagnostic.py
print_success "Created diagnostic.py"

# quick_test.py
cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""Quick test of Grok connection"""
import os
from dotenv import load_dotenv
load_dotenv()

from src.grok_client import GrokClient

client = GrokClient()
if client.test_connection():
    print("✅ Grok API connection successful!")
else:
    print("❌ Grok API connection failed. Check your .env file.")
EOF
chmod +x quick_test.py
print_success "Created quick_test.py"

# 9. Test connections (if not skipped)
if [ "$SKIP_TESTS" = false ]; then
    print_header "Testing Connections"

    # Test Grok API if .env is configured
    if grep -q "your_actual_key_here" .env 2>/dev/null; then
        print_warning "Skipping API test - .env not configured yet"
    else
        print_status "Testing Grok API..."
        if "$VENV_PY" quick_test.py 2>/dev/null; then
            print_success "Grok API working"
        else
            print_warning "Grok API test failed - check your XAI_API_KEY"
        fi
    fi

    # Test Docker
    print_status "Testing Docker..."
    if docker run --rm hello-world >/dev/null 2>&1; then
        print_success "Docker working"
    else
        print_warning "Docker test failed"
    fi

    # Test Terminal-Bench wrapper
    print_status "Testing Terminal-Bench wrapper..."
    if command -v timeout >/dev/null 2>&1; then
        if timeout 5 "$VENV_DIR/bin/tb-safe" --version >/dev/null 2>&1; then
            print_success "Terminal-Bench wrapper working"
        else
            print_warning "Terminal-Bench may still be initializing"
        fi
    else
        "$VENV_DIR/bin/tb-safe" --version >/dev/null 2>&1 || print_warning "Terminal-Bench may still be initializing"
    fi
fi

# 10. Final summary
print_header "Setup Complete! 🎉"

echo -e "${GREEN}${BOLD}Environment is ready for Terminal-Bench evaluation${NC}\n"

# Status Summary
echo -e "${BOLD}Status Summary:${NC}"
echo -e "  ✓ Python ${GREEN}$("$VENV_PY" --version 2>&1 | cut -d' ' -f2)${NC}"
echo -e "  ✓ Docker ${GREEN}running${NC}"
echo -e "  ✓ terminal-bench ${GREEN}installed${NC}"

if [ -f ".env" ]; then
    if grep -q "your_actual_key_here" .env 2>/dev/null; then
        echo -e "  ${YELLOW}⚠ .env needs configuration${NC}"
    else
        echo -e "  ✓ .env ${GREEN}configured${NC}"
    fi
fi

# Available commands
echo -e "\n${BOLD}Available Commands:${NC}"
echo -e "  ${BLUE}source venv/bin/activate${NC}       # Activate environment"
echo -e "  ${BLUE}$VENV_PY quick_test.py${NC}        # Test Grok API"
echo -e "  ${BLUE}$VENV_PY diagnostic.py${NC}        # Run full diagnostics"
echo -e "  ${BLUE}$VENV_PY run.py --test${NC}        # Quick benchmark test"
echo -e "  ${BLUE}$VENV_PY run.py --help${NC}        # See all options"

# Terminal-Bench specific commands
echo -e "\n${BOLD}Terminal-Bench Commands:${NC}"
echo -e "  ${BLUE}$VENV_DIR/bin/tb-safe --help${NC}  # Safe tb wrapper (handles timeouts)"
echo -e "  ${BLUE}$VENV_DIR/bin/tb-direct --help${NC}# Direct Python module access"
echo -e "  ${BLUE}python -m terminal_bench${NC}      # Works while venv is active"

# Next steps
echo -e "\n${BOLD}Next Steps:${NC}"

step_num=1
if grep -q "your_actual_key_here" .env 2>/dev/null; then
    echo -e "  ${YELLOW}${step_num}. Configure your API key:${NC}"
    echo -e "     - Get key from https://console.x.ai"
    echo -e "     - Edit .env and replace 'your_actual_key_here'"
    ((step_num++))
fi

echo -e "  ${GREEN}${step_num}. Test your setup:${NC}"
echo -e "     $VENV_PY diagnostic.py"
((step_num++))

echo -e "  ${GREEN}${step_num}. Run a quick test:${NC}"
echo -e "     $VENV_PY run.py --test"
((step_num++))

echo -e "  ${GREEN}${step_num}. Run the benchmark:${NC}"
echo -e "     $VENV_PY run.py --model grok-2-1212 --n-concurrent 4"

# Important notes
echo -e "\n${BOLD}Important Notes:${NC}"
echo -e "  • Terminal-Bench downloads datasets on first use (can take time)"
echo -e "  • If 'tb' command times out, use 'python -m terminal_bench' or tb-safe/tb-direct"
echo -e "  • Docker must be running for all benchmarks"
echo -e "  • Results are saved to results/ directory"

echo -e "\n${GREEN}${BOLD}Setup complete! Ready to benchmark Grok on Terminal-Bench. 🚀${NC}\n"
