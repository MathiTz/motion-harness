#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Motion Harness — Native Installer
#
# Sets up a Python venv, installs dependencies, and creates a `motion` shell
# command so you can launch the TUI from anywhere with a single word.
#
# Usage:  ./install.sh [options]
#   --shell NAME            configure this shell instead of auto-detecting
#   --link                  also symlink `motion` into a PATH directory (works in any shell)
#   --shell-only            only (re)create the launcher + shell integration
#   --no-shell              install everything except the shell integration
#   --uninstall             remove what this installer added (shell blocks and the symlink)
#   -h, --help              show this help
#
# Shells: fish, zsh, bash, ksh, tcsh and csh get a `motion` alias/function in their
# config file. sh/dash, nushell, elvish, xonsh, PowerShell and anything unknown get a
# symlink in ~/.local/bin instead (works in every shell) plus the PATH line to add.
#
# Environment: MOTION_SHELL (same as --shell), MOTION_BIN_DIR (launcher dir),
# MOTION_LINK_DIR (where the symlink goes).
# ──────────────────────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
MARK_START="# motion-harness-start"
MARK_END="# motion-harness-end"

usage() {
    cat << 'USAGE_EOF'
Usage: ./install.sh [options]
  --shell NAME            configure this shell instead of auto-detecting
                          (fish zsh bash ksh tcsh csh sh dash nu elvish xonsh pwsh)
  --link                  also symlink `motion` into a PATH directory (works in any shell)
  --shell-only            only (re)create the launcher + shell integration
  --no-shell              install everything except the shell integration
  --uninstall             remove what this installer added (shell blocks and the symlink)
  -h, --help              show this help

fish, zsh, bash, ksh, tcsh and csh get an alias/function in their config file. sh/dash,
nushell, elvish, xonsh, PowerShell and unknown shells get a symlink in ~/.local/bin
(works in every shell) and the PATH line to add if it isn't on PATH yet.

Environment: MOTION_SHELL (same as --shell), MOTION_BIN_DIR (launcher directory),
MOTION_LINK_DIR (symlink directory).
USAGE_EOF
}

# ── shell detection ───────────────────────────────────────────────────────────

# "-zsh", "/usr/bin/fish", "FISH", "ksh93", "dash", "nushell" -> a canonical name
# (fish zsh bash ksh tcsh csh sh nu elvish xonsh pwsh); anything else -> "".
normalize_shell() {
    local name
    name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    name="${name##*/}"
    name="${name#-}"
    case "$name" in
        fish|zsh|bash|tcsh|csh|elvish|xonsh) printf '%s' "$name" ;;
        ksh|ksh93|mksh|pdksh|oksh) printf 'ksh' ;;
        sh|dash|ash|busybox) printf 'sh' ;;
        nu|nushell) printf 'nu' ;;
        pwsh|powershell) printf 'pwsh' ;;
        *) printf '' ;;
    esac
}

# How we integrate: "block" = write an alias/function into the shell's config file;
# "link" = symlink the launcher into a PATH directory (needs no shell-specific syntax).
shell_method() {
    case "$1" in
        fish|zsh|bash|ksh|tcsh|csh) printf 'block' ;;
        *) printf 'link' ;;
    esac
}

# The shell the user is actually typing in: the nearest shell among this
# process's ancestors. $SHELL is only the *login* shell - e.g. a fish session
# started from a bash login still reports /bin/bash, which is how the alias used
# to end up in ~/.bashrc instead of fish's config.
shell_from_ancestors() {
    # MOTION_ANCESTOR_PID is a test hook; normally we start from our parent.
    local pid="${MOTION_ANCESTOR_PID:-$PPID}" comm found depth=0
    while [ -n "$pid" ] && [ "$pid" -gt 1 ] 2>/dev/null && [ "$depth" -lt 6 ]; do
        comm="$(ps -o comm= -p "$pid" 2>/dev/null | head -n 1)"
        found="$(normalize_shell "$comm")"
        if [ -n "$found" ]; then
            printf '%s' "$found"
            return 0
        fi
        pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
        depth=$((depth + 1))
    done
    return 1
}

# Sets DETECTED_SHELL and DETECT_REASON. Order: explicit choice, the shell the
# script was started from, the login shell.
detect_shell() {
    local requested="${1:-${MOTION_SHELL:-}}" login found
    DETECTED_SHELL=""
    DETECT_REASON=""
    login="$(normalize_shell "${SHELL:-}")"
    if [ -n "$requested" ]; then
        found="$(normalize_shell "$requested")"
        if [ -z "$found" ]; then
            echo "❌ Unsupported shell '$requested' (supported: fish zsh bash ksh tcsh csh sh dash nu elvish xonsh pwsh)." >&2
            return 2
        fi
        DETECTED_SHELL="$found"
        DETECT_REASON="requested explicitly"
    elif found="$(shell_from_ancestors)" && [ -n "$found" ]; then
        DETECTED_SHELL="$found"
        DETECT_REASON="the shell you ran this from"
        if [ -n "$login" ] && [ "$login" != "$found" ]; then
            DETECT_REASON="$DETECT_REASON; your login shell is $login"
        fi
    elif [ -n "$login" ]; then
        DETECTED_SHELL="$login"
        DETECT_REASON="login shell (\$SHELL)"
    fi
    return 0
}

rc_file_for() {
    case "$1" in
        fish) printf '%s' "${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish" ;;
        zsh)  printf '%s' "${ZDOTDIR:-$HOME}/.zshrc" ;;
        bash)
            # macOS terminals start bash as a login shell, which reads
            # ~/.bash_profile and not ~/.bashrc.
            if [ "$(uname -s)" = "Darwin" ]; then printf '%s' "$HOME/.bash_profile"; else printf '%s' "$HOME/.bashrc"; fi
            ;;
        ksh)  printf '%s' "${ENV:-$HOME/.kshrc}" ;;
        tcsh)
            # tcsh reads ~/.tcshrc, falling back to ~/.cshrc if that is all there is.
            if [ ! -f "$HOME/.tcshrc" ] && [ -f "$HOME/.cshrc" ]; then printf '%s' "$HOME/.cshrc"; else printf '%s' "$HOME/.tcshrc"; fi
            ;;
        csh)  printf '%s' "$HOME/.cshrc" ;;
    esac
}

# ── editing the config file (portable: no in-place sed, whose syntax differs on BSD/GNU) ─

# Single-quote for sh/bash/zsh: it'd -> 'it'\''d
sh_quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
# Single-quote for fish: backslash and quote are escaped with a backslash.
fish_quote() { printf "'%s'" "$(printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e "s/'/\\\\'/g")"; }

# A "plain" path needs no quoting at all.
plain_path() { case "$1" in *[!A-Za-z0-9_./+:@%,=-]*) return 1 ;; esac; return 0; }

# The whole right-hand side of a bash/zsh/ksh alias for the launcher path.
# An alias value is re-parsed every time it is used, so a path containing spaces or
# quotes has to be quoted *inside* the value (alias motion='"/a b/c"' style), not just
# when it is written. (Plain `alias motion='/a b/c'` expands to two words.)
sh_alias_token() {
    if plain_path "$1"; then sh_quote "$1"; else sh_quote "$(sh_quote "$1")"; fi
}

# Same for csh/tcsh: alias motion '"/a b/c"'. csh cannot escape ' inside '...', treats !
# as history and still expands $ and backticks inside "...", so such paths return 1 and
# the caller falls back to the PATH symlink.
csh_quote() {
    if plain_path "$1"; then printf "'%s'" "$1"; return 0; fi
    case "$1" in
        *"'"*|*'"'*|*'!'*|*'$'*|*'`'*|*'\\'*|*'
'*) return 1 ;;
    esac
    printf "'\"%s\"'" "$1"
}

# 0 if every start marker is followed by exactly one end marker, in order.
blocks_well_formed() {
    awk -v s="$MARK_START" -v e="$MARK_END" '
        { line = $0; sub(/[ \t\r]+$/, "", line) }
        line == s { if (open) bad = 1; open = 1; next }
        line == e { if (!open) bad = 1; open = 0; next }
        END { exit (bad || open) ? 1 : 0 }
    ' "$1"
}

# Remove every motion block (and the one blank separator line before it).
# Refuses to touch a file whose markers are unbalanced: the old
# `sed '/start/,/end/d'` deleted everything after a lone start marker.
remove_blocks() {
    local file="$1" tmp
    [ -f "$file" ] || return 0
    grep -q -F "# motion-harness" "$file" 2>/dev/null || return 0
    if ! blocks_well_formed "$file"; then
        echo "⚠️  $file has unbalanced '# motion-harness' markers; leaving it untouched." >&2
        echo "   Remove the motion lines by hand, then re-run." >&2
        return 1
    fi
    tmp="$(mktemp "${TMPDIR:-/tmp}/motion-install.XXXXXX")"
    awk -v s="$MARK_START" -v e="$MARK_END" '
        { line = $0; sub(/[ \t\r]+$/, "", line) }
        skip { if (line == e) skip = 0; next }
        line == s { skip = 1; havepend = 0; next }
        havepend { print pend; havepend = 0 }
        $0 == "" { pend = $0; havepend = 1; next }
        { print }
        END { if (havepend) print pend }
    ' "$file" > "$tmp"
    # Write *through* the file rather than replacing it, so a config that is
    # a symlink into a dotfiles repo stays a symlink.
    cat "$tmp" > "$file"
    rm -f "$tmp"
}

install_shell_integration() {
    local shell_type="$1" config_file="$2"
    if { [ "$shell_type" = tcsh ] || [ "$shell_type" = csh ]; } && ! csh_quote "$WRAPPER" > /dev/null; then
        echo "⚠️  The launcher path contains a quote or '!', which $shell_type can't alias safely." >&2
        return 3
    fi
    mkdir -p "$(dirname "$config_file")"
    touch "$config_file"
    remove_blocks "$config_file" || return 1

    # Keep the user's last line intact if it has no trailing newline.
    if [ -s "$config_file" ] && [ -n "$(tail -c 1 "$config_file")" ]; then
        echo "" >> "$config_file"
    fi
    {
        echo ""
        echo "$MARK_START"
        case "$shell_type" in
            fish)
                echo "function motion"
                echo "    $(fish_quote "$WRAPPER") \$argv"
                echo "end" ;;
            tcsh|csh) echo "alias motion $(csh_quote "$WRAPPER")" ;;
            *)        echo "alias motion=$(sh_alias_token "$WRAPPER")" ;;
        esac
        echo "$MARK_END"
    } >> "$config_file"
    echo "📝 Added the 'motion' command to $config_file"
}

on_path() { case ":$PATH:" in *":$1:"*) return 0 ;; *) return 1 ;; esac; }

# Directory for the `motion` symlink: an existing user bin dir that is already on PATH,
# else ~/.local/bin (created).
pick_link_dir() {
    local d
    if [ -n "${MOTION_LINK_DIR:-}" ]; then printf '%s' "$MOTION_LINK_DIR"; return; fi
    for d in "$HOME/.local/bin" "$HOME/bin" "$HOME/.bin"; do
        if [ -d "$d" ] && on_path "$d"; then printf '%s' "$d"; return; fi
    done
    printf '%s' "$HOME/.local/bin"
}

# Symlink the launcher into a PATH directory. Never replaces something that isn't ours.
install_link() {
    local dir link current
    dir="$(pick_link_dir)"
    link="$dir/motion"
    mkdir -p "$dir"
    if [ -e "$link" ] || [ -L "$link" ]; then
        current="$(readlink "$link" 2>/dev/null || true)"
        if [ -L "$link" ] && { [ "$current" = "$WRAPPER" ] || [ ! -e "$link" ]; }; then
            rm -f "$link"
        else
            echo "⚠️  $link already exists and is not this installer's link; leaving it alone." >&2
            LINK_DIR="$dir"
            return 3
        fi
    fi
    ln -s "$WRAPPER" "$link"
    LINK_DIR="$dir"
    echo "🔗 Linked $link -> $WRAPPER"
}

# The line that puts a directory on PATH, in the given shell's syntax.
path_line() {
    local shell_type="$1" dir="$2"
    case "$shell_type" in
        fish)   echo "fish_add_path $dir" ;;
        tcsh|csh) echo "set path = ($dir \$path)" ;;
        nu)     echo "\$env.PATH = (\$env.PATH | prepend '$dir')" ;;
        elvish) echo "set paths = [$dir \$@paths]" ;;
        xonsh)  echo "\$PATH.insert(0, '$dir')" ;;
        pwsh)   echo "\$env:PATH = \"$dir\" + [IO.Path]::PathSeparator + \$env:PATH" ;;
        *)      echo "export PATH=\"$dir:\$PATH\"" ;;
    esac
}

path_config_hint() {
    case "$1" in
        fish)   echo "~/.config/fish/config.fish" ;;
        tcsh|csh) echo "~/.tcshrc or ~/.cshrc" ;;
        nu)     echo "config.nu (run \`config nu\`)" ;;
        elvish) echo "~/.config/elvish/rc.elv" ;;
        xonsh)  echo "~/.xonshrc" ;;
        pwsh)   echo "your PowerShell profile (\$PROFILE)" ;;
        sh)     echo "~/.profile" ;;
        *)      echo "your shell's startup file" ;;
    esac
}

reload_hint() {
    case "$1" in
        fish) echo "source $(rc_file_for fish)     # or open a new terminal" ;;
        zsh)  echo "source $(rc_file_for zsh)" ;;
        bash) echo "source $(rc_file_for bash)" ;;
        ksh)  echo ". $(rc_file_for ksh)" ;;
        tcsh) echo "source $(rc_file_for tcsh)" ;;
        csh)  echo "source $(rc_file_for csh)" ;;
        *)    echo "open a new terminal" ;;
    esac
}

# ── main ──────────────────────────────────────────────────────────────────────

do_uninstall() {
    local removed=0 f dir link current wrapper
    for f in "$(rc_file_for fish)" "$(rc_file_for zsh)" "$HOME/.bashrc" "$HOME/.bash_profile" \
             "$(rc_file_for ksh)" "$HOME/.tcshrc" "$HOME/.cshrc"; do
        if [ -f "$f" ] && grep -q -F "$MARK_START" "$f" 2>/dev/null; then
            if remove_blocks "$f"; then
                echo "🧹 Removed the motion block from $f"
                removed=$((removed + 1))
            fi
        fi
    done
    # Remove our symlink (only ours: it must point at this launcher).
    wrapper="${MOTION_BIN_DIR:-$REPO_DIR/bin}/motion"
    for dir in "${MOTION_LINK_DIR:-}" "$HOME/.local/bin" "$HOME/bin" "$HOME/.bin"; do
        [ -n "$dir" ] || continue
        link="$dir/motion"
        if [ -L "$link" ]; then
            current="$(readlink "$link" 2>/dev/null || true)"
            if [ "$current" = "$wrapper" ]; then
                rm -f "$link"
                echo "🧹 Removed the symlink $link"
                removed=$((removed + 1))
            fi
        fi
    done
    [ "$removed" -eq 0 ] && echo "Nothing to remove: no motion block or symlink found."
    return 0
}

main() {
    set -e
    local shell_arg="" shell_only=0 no_shell=0 uninstall=0 want_link=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --shell)      shell_arg="${2:-}"; [ -n "$shell_arg" ] || { echo "❌ --shell needs a value" >&2; return 2; }; shift 2 ;;
            --shell=*)    shell_arg="${1#--shell=}"; shift ;;
            --link)       want_link=1; shift ;;
            --shell-only) shell_only=1; shift ;;
            --no-shell)   no_shell=1; shift ;;
            --uninstall)  uninstall=1; shift ;;
            -h|--help)    usage; return 0 ;;
            *) echo "❌ Unknown option: $1 (try --help)" >&2; return 2 ;;
        esac
    done

    if [ "$uninstall" -eq 1 ]; then
        do_uninstall
        return 0
    fi

    echo "🚀 Installing Motion Harness..."
    echo "   Repo: $REPO_DIR"

    if [ "$shell_only" -eq 0 ]; then
        # ── 1. Python ─────────────────────────────────────────────────────────
        if ! command -v python3 &> /dev/null; then
            echo "❌ Error: python3 not found. Install Python 3.11+ first."
            return 1
        fi
        echo "   Python: $(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"

        # ── 2. Virtual environment ────────────────────────────────────────────
        if [ ! -d "$VENV_DIR" ]; then
            echo "📦 Creating virtual environment..."
            python3 -m venv "$VENV_DIR"
        else
            echo "✅ Virtual environment exists"
        fi
        echo "📥 Installing dependencies..."
        "$VENV_DIR/bin/pip" install -q --upgrade pip
        "$VENV_DIR/bin/pip" install -q -r "$REPO_DIR/requirements.txt"

        # ── 3. Config ─────────────────────────────────────────────────────────
        if [ ! -f "$REPO_DIR/config.yml" ]; then
            if [ -f "$REPO_DIR/config.example.yml" ]; then
                echo "📋 Copying config.example.yml → config.yml"
                cp "$REPO_DIR/config.example.yml" "$REPO_DIR/config.yml"
                echo "   Edit config.yml and .env with your API keys."
            fi
        else
            echo "✅ config.yml exists"
        fi
    fi

    # ── 4. Create the wrapper script ──────────────────────────────────────────
    local bin_dir="${MOTION_BIN_DIR:-$REPO_DIR/bin}"
    WRAPPER="$bin_dir/motion"
    mkdir -p "$bin_dir"
    cat > "$WRAPPER" << WRAPPER_EOF
#!/bin/bash
# Motion Harness launcher — created by install.sh
#
# Intentionally does NOT cd into the repo: the harness's own state
# (config.yml, .env, memory DB, synthesized skills) is resolved relative to
# REPO_DIR by the Python code itself, while the *workspace* the agent reads
# and writes files in is whatever directory you run \`motion\` from. Point
# it at any project by cd-ing there first, just like any other CLI tool.
REPO_DIR="$REPO_DIR"
VENV_DIR="$VENV_DIR"
export PYTHONPATH="\$REPO_DIR"
exec "\$VENV_DIR/bin/python" "\$REPO_DIR/main.py" "\$@"
WRAPPER_EOF
    chmod +x "$WRAPPER"
    echo "✅ Launcher: $WRAPPER"

    # ── 5. Shell integration ──────────────────────────────────────────────────
    local hint="open a new terminal" method="" linked=0 note=""
    if [ "$no_shell" -eq 1 ]; then
        echo "⏭  Skipping shell integration. Run $WRAPPER directly, or add an alias yourself."
    else
        detect_shell "$shell_arg" || return $?
        if [ -z "$DETECTED_SHELL" ]; then
            echo "🐚 Could not tell which shell you use: falling back to a PATH symlink, which works in any shell."
            method="link"
        else
            echo "🐚 Shell: $DETECTED_SHELL ($DETECT_REASON)"
            method="$(shell_method "$DETECTED_SHELL")"
        fi

        if [ "$method" = "block" ]; then
            local rc rc_status=0
            rc="$(rc_file_for "$DETECTED_SHELL")"
            install_shell_integration "$DETECTED_SHELL" "$rc" || rc_status=$?
            if [ "$rc_status" -eq 0 ]; then
                hint="$(reload_hint "$DETECTED_SHELL")"
                echo "   Wrong shell? Re-run with --shell NAME (and --uninstall removes what was added)."
            elif [ "$rc_status" -eq 3 ]; then
                method="link"           # e.g. csh can't quote this path: use the symlink instead
            else
                return "$rc_status"
            fi
        fi

        if [ "$method" = "link" ] || [ "$want_link" -eq 1 ]; then
            install_link && linked=1 || true
            if [ "$linked" -eq 1 ]; then
                if on_path "$LINK_DIR"; then
                    [ "$method" = "link" ] && hint="open a new terminal (or just run: motion)"
                else
                    note="Add $LINK_DIR to your PATH — put this in $(path_config_hint "${DETECTED_SHELL:-sh}"):
       $(path_line "${DETECTED_SHELL:-sh}" "$LINK_DIR")"
                    [ "$method" = "link" ] && hint="add $LINK_DIR to PATH (see below), then open a new terminal"
                fi
            fi
        fi
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "✨ Motion Harness installed!"
    echo ""
    echo "   Quick start:"
    echo "     $hint"
    echo "     motion                                    # Launch TUI"
    echo "     motion --provider ollama-cloud/glm-5.2    # With specific model"
    echo "     motion -p \"summarize README.md\"           # Headless one-shot"
    echo "     motion --list                             # List providers"
    echo ""
    [ -n "$note" ] && printf '\n   %s\n' "$note"
    echo ""
    echo "   Config: $REPO_DIR/config.yml"
    echo "   Venv:   $VENV_DIR"
    echo "───────────────────────────────────────────────────────────────"
}

# Run only when executed (or piped into bash), not when sourced by the tests.
if [ -z "${BASH_SOURCE[0]:-}" ] || [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
