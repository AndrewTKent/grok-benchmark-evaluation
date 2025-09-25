#!/bin/bash

#####################################################################
#  Setup Script for Grok Terminal-Bench Evaluation                  #
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

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

check_command() {
    if command -v $1 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_python_version() {
    if check_command python3; then
        version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        required=$1
        if [ "$(printf '%s\n' "$required" "$version" | sort -V | head -n1)" = "$required" ]; then
            return 0
        fi
    fi
    return 1
}

# Parse arguments
FORCE_REINSTALL=false
SKIP_TESTS=false
CLONE_TBENCH=true  # Default to true

while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_REINSTALL=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --no-clone-tbench)
            CLONE_TBENCH=false
            shift
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo "Options:"
            echo "  -f, --force         Force reinstall all components"
            echo "  --skip-tests        Skip connection tests"
            echo "  --no-clone-tbench   Skip cloning terminal-bench repository"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "Terminal-Bench (T-Bench) Environment Setup"

# ASCII art for T-Bench
echo -e "${BLUE}"
cat << 'EOF'
  _____                   _             _     ____                  _     
 |_   _|__ _ __ _ __ ___ (_)_ __   __ _| |   | __ )  ___ _ __   ___| |__  
   | |/ _ \ '__| '_ ` _ \| | '_ \ / _` | |   |  _ \ / _ \ '_ \ / __| '_ \ 
   | |  __/ |  | | | | | | | | | | (_| | |   | |_) |  __/ | | | (__| | | |
   |_|\___|_|  |_| |_| |_|_|_| |_|\__,_|_|   |____/ \___|_| |_|\___|_| |_|
EOF
echo -e "${NC}"

# 1. Check system dependencies
print_header "Checking System Dependencies"

# Check Python
if check_python_version $PYTHON_MIN_VERSION; then
    version=$(python3 --version)
    print_success "Python installed: $version"
else
    print_error "Python $PYTHON_MIN_VERSION+ is required"
    exit 1
fi

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
    
    # Check if Docker is running
    if docker info > /dev/null 2>&1; then
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

# Check for uv (optional but recommended for T-Bench)
if check_command uv; then
    print_success "uv is installed (recommended for T-Bench)"
else
    print_warning "uv not found. Installing uv is recommended..."
    echo "To install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Continuing with pip..."
fi

# 2. Setup Python virtual environment
print_header "Python Virtual Environment"

if [ -d "$VENV_DIR" ] && [ ! "$FORCE_REINSTALL" = true ]; then
    print_status "Virtual environment exists, activating..."
    source $VENV_DIR/bin/activate
    
    if [ "$VIRTUAL_ENV" != "" ]; then
        print_success "Virtual environment activated"
    else
        print_warning "Failed to activate. Recreating..."
        rm -rf $VENV_DIR
        python3 -m venv $VENV_DIR
        source $VENV_DIR/bin/activate
        print_success "Virtual environment recreated"
    fi
else
    if [ "$FORCE_REINSTALL" = true ] && [ -d "$VENV_DIR" ]; then
        print_status "Force reinstall: Removing existing venv..."
        rm -rf $VENV_DIR
    fi
    
    print_status "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    print_success "Virtual environment created and activated"
fi

# Upgrade pip
print_status "Upgrading pip..."
pip install --quiet --upgrade pip

# 3. Clone/Update Terminal-Bench repository FIRST
if [ "$CLONE_TBENCH" = true ]; then
    print_header "Terminal-Bench Repository"
    
    if [ -d "$TBENCH_DIR" ]; then
        if [ "$FORCE_REINSTALL" = true ]; then
            print_status "Force reinstall: Re-cloning terminal-bench..."
            rm -rf $TBENCH_DIR
            git clone $TBENCH_REPO $TBENCH_DIR
            print_success "Terminal-bench cloned fresh"
        else
            print_status "Terminal-bench exists. Checking for updates..."
            cd $TBENCH_DIR
            
            if [ -d ".git" ]; then
                git fetch origin -q
                LOCAL=$(git rev-parse @)
                REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "")
                
                if [ "$LOCAL" != "$REMOTE" ] && [ "$REMOTE" != "" ]; then
                    print_warning "Updates available for terminal-bench"
                    read -p "Update to latest version? (y/n) " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        git pull origin main
                        print_success "Updated to latest version"
                    fi
                else
                    current_commit=$(git rev-parse --short HEAD)
                    print_success "Terminal-bench is up to date (commit: $current_commit)"
                fi
            fi
            cd ..
        fi
    else
        print_status "Cloning terminal-bench repository..."
        git clone $TBENCH_REPO $TBENCH_DIR
        print_success "Terminal-bench cloned"
    fi
fi

# 4. Install Terminal-Bench and dependencies
print_header "Installing Dependencies"

# Install terminal-bench package (from cloned repo if available)
if [ -d "$TBENCH_DIR" ] && [ -f "$TBENCH_DIR/setup.py" -o -f "$TBENCH_DIR/pyproject.toml" ]; then
    print_status "Installing terminal-bench from cloned repository..."
    pip install -e $TBENCH_DIR/
    print_success "terminal-bench installed from local repository"
else
    print_status "Installing terminal-bench package from PyPI..."
    pip install --upgrade terminal-bench
    print_success "terminal-bench package installed from PyPI"
fi

# Check if tb command works
if pip show terminal-bench > /dev/null 2>&1; then
    print_success "terminal-bench package is available"
    
    # Try to get tb working
    if check_command tb; then
        print_success "tb CLI is available"
    else
        # Try to find tb in venv
        if [ -f "$VENV_DIR/bin/tb" ]; then
            print_success "tb CLI found in venv"
            export PATH="$PWD/$VENV_DIR/bin:$PATH"
        else
            print_warning "tb CLI not in PATH. You may need to use: python -m terminal_bench"
        fi
    fi
else
    print_error "Failed to install terminal-bench"
fi

# Install additional requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    print_status "Installing additional requirements..."
    pip install -q -r requirements.txt
    print_success "Additional requirements installed"
else
    print_status "Installing common dependencies..."
    pip install requests python-dotenv pandas tqdm jsonlines numpy openai aiohttp
    print_success "Common dependencies installed"
fi

# 5. Test connections (if not skipped)
if [ "$SKIP_TESTS" = false ]; then
    print_header "Testing Connections"
    
    # Test tb CLI
    if check_command tb || [ -f "$VENV_DIR/bin/tb" ]; then
        print_status "Testing Terminal-Bench CLI..."
        if tb --help > /dev/null 2>&1 || $VENV_DIR/bin/tb --help > /dev/null 2>&1; then
            print_success "Terminal-Bench CLI is working"
        fi
    fi
    
    # Test Docker
    print_status "Testing Docker connection..."
    if docker run --rm hello-world > /dev/null 2>&1; then
        print_success "Docker is working correctly"
    else
        print_warning "Docker test failed - check Docker installation"
    fi
fi

# 6. Final summary
print_header "Setup Complete! 🎉"

echo -e "${GREEN}${BOLD}Environment is ready for Terminal-Bench evaluation${NC}\n"

# Show status summary
echo -e "${BOLD}Status Summary:${NC}"
echo -e "  ✓ Python ${GREEN}$(python3 --version 2>&1 | cut -d' ' -f2)${NC}"
echo -e "  ✓ Docker ${GREEN}running${NC}"
echo -e "  ✓ terminal-bench ${GREEN}installed${NC}"
if [ "$CLONE_TBENCH" = true ]; then
    echo -e "  ✓ Repository ${GREEN}cloned${NC}"
else
    echo -e "  ✓ Repository ${YELLOW}not cloned${NC}"
fi

# Check what needs to be done
echo -e "\n${BOLD}Next Steps:${NC}"

step_num=1
if [ ! -f ".env" ]; then
    echo -e "  ${YELLOW}${step_num}. Create .env file with your Grok API key:${NC}"
    echo -e "     - Get key from https://console.x.ai"
    echo -e "     - Add XAI_API_KEY=your_actual_key_here to .env"
    ((step_num++))
fi

echo -e "  ${GREEN}${step_num}. Test your setup:${NC}"
echo -e "     source venv/bin/activate"
if [ -f "quick_test.py" ]; then
    echo -e "     python quick_test.py"
fi
((step_num++))

if [ "$CLONE_TBENCH" = true ]; then
    echo -e "  ${GREEN}${step_num}. Explore Terminal-Bench:${NC}"
    echo -e "     - View tasks: ls terminal-bench/tasks/"
    echo -e "     - Read docs: cat terminal-bench/README.md"
    ((step_num++))
fi

echo -e "  ${GREEN}${step_num}. Run Terminal-Bench with Grok:${NC}"
echo -e "     - Example: python run.py --model grok-3 --dataset terminal-bench-core==0.1.1"
echo -e "     - See README.md for more options"

# Helpful commands
echo -e "\n${BOLD}Useful Commands:${NC}"
echo -e "  ${BLUE}source venv/bin/activate${NC}     # Activate environment"
echo -e "  ${BLUE}tb --help${NC}                    # Terminal-Bench help"
if [ -f "quick_test.py" ]; then
    echo -e "  ${BLUE}python quick_test.py${NC}         # Test Grok connection"
fi
echo -e "  ${BLUE}./setup.sh --force${NC}           # Reinstall everything"
if [ "$CLONE_TBENCH" = true ]; then
    echo -e "  ${BLUE}ls terminal-bench/tasks/${NC}     # View available tasks"
fi

# Important notes
echo -e "\n${BOLD}Important Notes:${NC}"
echo -e "  • Terminal-Bench runs tasks in Docker containers"
echo -e "  • The official tb CLI expects specific agent adapters"
echo -e "  • Results are saved to results/ directory"
if [ "$CLONE_TBENCH" = true ]; then
    echo -e "  • terminal-bench repository cloned for local development"
fi

echo -e "\n${GREEN}${BOLD}Ready to benchmark! 🚀${NC}\n"