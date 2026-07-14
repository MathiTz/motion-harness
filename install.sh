#!/bin/bash

# Motion Harness Installer
# This script automates the setup of the Motion Harness Docker distribution.

echo "🚀 Installing Motion Harness..."

# 1. Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed. Please install Docker Desktop first."
    exit 1
fi

# 2. Build the Docker Image
echo "📦 Building the MotionL Harness image (this may take a few minutes)..."
docker build -t motion-harness .

# 3. Create the global alias
# Detect shell to determine the correct config file
SHELL_FILE=""
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_FILE="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_FILE="$HOME/.bashrc"
elif [[ "$SHELL" == *"fish"* ]]; then
    # Fish uses a different syntax for aliases/functions
    SHELL_FILE="$HOME/.config/fish/config.fish"
fi

if [ -z "$SHELL_FILE" ]; then
    echo "⚠️ Could not detect shell config file. You will need to add the alias manually."
    echo "Command: alias motion='docker run -it --rm -v \"\$(pwd):/app\" -v \"\$HOME/.motion_harness:/root/.hermes\" motion-harness'"
    exit 0
fi

# Add the alias to the config file
# We use a function for fish and an alias for bash/zsh
if [[ "$SHELL_FILE" == *.fish ]]; then
    ALIAS_CMD="function motion; docker run -it --rm -v \"(pwd):/app\" -v \"$HOME/.motion_harness:/root/.hermes\" motion-harness; end"
else
    ALIAS_CMD="alias motion='docker run -it --rm -v \"$(pwd):/app\" -v \"$HOME/.motion_harness:/root/.hermes\" motion-harness'"
fi

# Check if alias already exists to avoid duplicates
if grep -q "alias motion=" "$SHELL_FILE" 2>/dev/null || grep -q "function motion" "$SHELL_FILE" 2>/dev/null; then
    echo "✅ Alias 'motion' already exists in $SHELL_FILE."
else
    echo "📝 Adding 'motion' alias to $SHELL_FILE..."
    echo "$ALIAS_CMD" >> "$SHELL_FILE"
fi

echo "-----------------------------------------------------------------"
echo "✨ Installation Complete!"
echo "1. Please restart your terminal or run: source $SHELL_FILE"
echo "2. You can now launch the harness from any project folder using:"
echo "   motion"
echo "-----------------------------------------------------------------"
