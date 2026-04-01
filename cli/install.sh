#!/usr/bin/env bash
# OpenPI CLI Installer
# Usage: curl -fsSL https://get.openpi.dev/install.sh | sh
set -euo pipefail

OPENPI_HOME="${OPENPI_HOME:-$HOME/.openpi}"
VENV_DIR="$OPENPI_HOME/venv"
BIN_DIR="${HOME}/.local/bin"
MIN_PYTHON="3.10"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}==>${NC} $*"; }
ok()    { echo -e "${GREEN}==>${NC} $*"; }
warn()  { echo -e "${YELLOW}==>${NC} $*"; }
err()   { echo -e "${RED}==>${NC} $*" >&2; }

banner() {
    echo ""
    echo -e "${BOLD}  ┌─────────────────────────────────────┐${NC}"
    echo -e "${BOLD}  │         OpenPI CLI Installer         │${NC}"
    echo -e "${BOLD}  │   AI-Powered Research Pipeline       │${NC}"
    echo -e "${BOLD}  └─────────────────────────────────────┘${NC}"
    echo ""
}

# Check if a command exists
has() { command -v "$1" >/dev/null 2>&1; }

# Compare Python versions (returns 0 if $1 >= $2)
version_gte() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Find a suitable Python >= 3.10
find_python() {
    for cmd in python3.12 python3.11 python3.10 python3 python; do
        if has "$cmd"; then
            local ver
            ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
            if [ -n "$ver" ] && version_gte "$ver" "$MIN_PYTHON"; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        *)       echo "unknown" ;;
    esac
}

main() {
    banner

    local os
    os=$(detect_os)
    info "Detected OS: $os ($(uname -m))"

    # Find Python
    local python_cmd
    if python_cmd=$(find_python); then
        local python_ver
        python_ver=$("$python_cmd" --version 2>&1)
        ok "Found $python_ver ($python_cmd)"
    else
        err "Python >= $MIN_PYTHON not found."
        echo ""
        echo "  Install Python first:"
        case "$os" in
            macos)  echo "    brew install python@3.12" ;;
            linux)  echo "    sudo apt install python3.12 python3.12-venv  # Debian/Ubuntu"
                    echo "    sudo dnf install python3.12                  # Fedora/RHEL" ;;
        esac
        echo ""
        echo "  Or use pyenv: https://github.com/pyenv/pyenv"
        exit 1
    fi

    # Create OpenPI home
    info "Creating $OPENPI_HOME..."
    mkdir -p "$OPENPI_HOME"

    # Create venv
    if [ -d "$VENV_DIR" ]; then
        warn "Existing venv found at $VENV_DIR"
        read -rp "  Recreate? [y/N] " ans
        if [[ "$ans" =~ ^[Yy] ]]; then
            rm -rf "$VENV_DIR"
            info "Creating virtual environment..."
            "$python_cmd" -m venv "$VENV_DIR"
        fi
    else
        info "Creating virtual environment..."
        "$python_cmd" -m venv "$VENV_DIR"
    fi

    # Upgrade pip
    info "Upgrading pip..."
    "$VENV_DIR/bin/pip" install --upgrade pip -q

    # Install openpi-cli
    info "Installing openpi-cli..."
    if [ -f "pyproject.toml" ] && grep -q "openpi-cli" pyproject.toml 2>/dev/null; then
        # Local install (developer mode)
        "$VENV_DIR/bin/pip" install -e . -q
    else
        # PyPI install
        "$VENV_DIR/bin/pip" install openpi-cli -q 2>/dev/null || {
            warn "openpi-cli not on PyPI yet. Installing from source..."
            "$VENV_DIR/bin/pip" install git+https://github.com/PierBeneventano/OpenPI.git#subdirectory=cli -q
        }
    fi

    # Create symlink
    mkdir -p "$BIN_DIR"
    ln -sf "$VENV_DIR/bin/openpi" "$BIN_DIR/openpi"

    # Check PATH
    if ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
        warn "$BIN_DIR is not in your PATH."
        echo ""
        echo "  Add it to your shell profile:"
        echo "    echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
        echo "    source ~/.bashrc"
        echo ""
    fi

    # Verify
    if has openpi || [ -x "$BIN_DIR/openpi" ]; then
        ok "openpi installed successfully!"
    else
        ok "openpi installed to $BIN_DIR/openpi"
    fi

    echo ""
    echo -e "${BOLD}  Next steps:${NC}"
    echo ""
    echo "    openpi setup            # Configure API keys and model"
    echo "    openpi doctor           # Verify your environment"
    echo "    openpi run \"question\"   # Start a research pipeline"
    echo ""
}

main "$@"
