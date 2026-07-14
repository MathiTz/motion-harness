#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Motion Harness — Native Installer
#
# Sets up a Python venv, installs dependencies, and creates a `motion` shell
# command so you can launch the TUI from anywhere with a single word.
#
# Usage:  chmod +x install.sh && ./install.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

echo "🚀 Installing Motion Harness..."
echo "   Repo: $REPO_DIR"

# ── 1. Python ─────────────────────────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Install Python 3.11+ first."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "   Python: $PY_VERSION"

# ── 2. Virtual environment ─────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "✅ Virtual environment exists"
fi

echo "📥 Installing dependencies..."
"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q -r "$REPO_DIR/requirements.txt"

# ── 3. Config ──────────────────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/config.yml" ]; then
    if [ -f "$REPO_DIR/config.example.yml" ]; then
        echo "📋 Copying config.example.yml → config.yml"
        cp "$REPO_DIR/config.example.yml" "$REPO_DIR/config.yml"
        echo "   Edit config.yml and .env with your API keys."
    fi
else
    echo "✅ config.yml exists"
fi

# ── 4. Create the wrapper script ───────────────────────────────────────────────
WRAPPER="$REPO_DIR/bin/motion"
mkdir -p "$REPO_DIR/bin"

cat > "$WRAPPER" << WRAPPER_EOF
#!/bin/bash
# Motion Harness launcher — created by install.sh
REPO_DIR="$REPO_DIR"
VENV_DIR="$REPO_DIR/.venv"
export PYTHONPATH="\$REPO_DIR"
cd "\$REPO_DIR"
exec "\$VENV_DIR/bin/python" "\$REPO_DIR/main.py" "\$@"
WRAPPER_EOF

chmod +x "$WRAPPER"
echo "✅ Wrapper script: $WRAPPER"

# ── 5. Shell integration ───────────────────────────────────────────────────────
DETECTED_SHELL="${SHELL##*/}"
echo "🐚 Detected shell: $DETECTED_SHELL"

install_shell_integration() {
    local shell_type="$1"
    local config_file="$2"

    # Remove old motion-harness block if present
    if grep -q "# motion-harness" "$config_file" 2>/dev/null; then
        # Remove everything between the markers (inclusive)
        sed -i '' '/# motion-harness-start/,/# motion-harness-end/d' "$config_file" 2>/dev/null || true
    fi

    echo "" >> "$config_file"
    echo "# motion-harness-start" >> "$config_file"
    if [ "$shell_type" = "fish" ]; then
        echo "function motion" >> "$config_file"
        echo "    \"$WRAPPER\" \$argv" >> "$config_file"
        echo "end" >> "$config_file"
    else
        echo "alias motion=\"$WRAPPER\"" >> "$config_file"
    fi
    echo "# motion-harness-end" >> "$config_file"
    echo "📝 Added 'motion' command to $config_file"
}

case "$DETECTED_SHELL" in
    fish)
        FISH_CONFIG="$HOME/.config/fish/config.fish"
        mkdir -p "$(dirname "$FISH_CONFIG")"
        install_shell_integration fish "$FISH_CONFIG"
        ;;
    zsh)
        install_shell_integration zsh "$HOME/.zshrc"
        ;;
    bash)
        install_shell_integration bash "$HOME/.bashrc"
        ;;
    *)
        echo "⚠️  Unknown shell: $DETECTED_SHELL"
        echo "   Add this to your shell config manually:"
        echo "   alias motion='$WRAPPER'"
        ;;
esac

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "───────────────────────────────────────────────────────────────"
echo "✨ Motion Harness installed!"
echo ""
echo "   Quick start:"
echo "     source ~/.config/fish/config.fish  # or .zshrc / .bashrc"
echo "     motion                                    # Launch TUI"
echo "     motion --provider ollama-cloud/glm-5.2    # With specific model"
echo "     motion --chat                             # Legacy REPL"
echo "     motion --list                             # List providers"
echo ""
echo "   Config: $REPO_DIR/config.yml"
echo "   Venv:   $VENV_DIR"
echo "───────────────────────────────────────────────────────────────"