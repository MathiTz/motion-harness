"""install.sh: shell detection, portable config editing, uninstall.

Everything runs against a throwaway HOME with --shell-only, so no test touches
the real shell configs, the venv, or the network.
"""
import os
import platform
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALL = REPO / "install.sh"
START, END = "# motion-harness-start", "# motion-harness-end"
# macOS bash reads ~/.bash_profile (login shell); Linux reads ~/.bashrc. The installer knows; so must the tests.
BASH_RC = ".bash_profile" if platform.system() == "Darwin" else ".bashrc"


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    h.mkdir()
    return h


def env_for(home: Path, tmp_path: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("XDG_CONFIG_HOME", "ZDOTDIR", "MOTION_SHELL", "SHELL")}
    env.update(HOME=str(home), MOTION_BIN_DIR=str(tmp_path / "bin"), SHELL="/bin/bash")
    env.update(extra)  # callers may override any of the above
    return env


def install(home, tmp_path, *args, **extra):
    return subprocess.run(["bash", str(INSTALL), "--shell-only", *args], capture_output=True, text=True,
                          env=env_for(home, tmp_path, **extra), cwd=str(tmp_path), timeout=60)


def fn(script: str, home: Path, tmp_path: Path, **extra):
    """Run a snippet with install.sh's functions loaded (sourcing does not run main)."""
    return subprocess.run(["bash", "-c", f'source "{INSTALL}"\n{script}'], capture_output=True, text=True,
                          env=env_for(home, tmp_path, **extra), timeout=30)


def count(path: Path, marker=START) -> int:
    return path.read_text().count(marker)


# ── shell detection ─────────────────────────────────────────────────────────

FAKE_PS = '''
ps() {
  local pid; pid="${@: -1}"
  case "$*" in
    *comm=*) case "$pid" in 101) echo "$C1";; 102) echo "$C2";; 103) echo "$C3";; esac ;;
    *ppid=*) case "$pid" in 101) echo 102;; 102) echo 103;; 103) echo 1;; esac ;;
  esac
}
MOTION_ANCESTOR_PID=101
'''


def detect(home, tmp_path, comms, login="/bin/bash", arg=""):
    c1, c2, c3 = (list(comms) + ["", "", ""])[:3]
    r = fn(f'{FAKE_PS}\ndetect_shell {arg}; echo "[$DETECTED_SHELL] [$DETECT_REASON]"', home, tmp_path,
           C1=c1, C2=c2, C3=c3, SHELL=login)
    line = r.stdout.strip().splitlines()[-1]
    shell, reason = line[1:].split("] [", 1)
    return shell, reason.rstrip("]"), r


def test_the_running_shell_beats_the_login_shell(home, tmp_path):
    """Bug 1: $SHELL said bash inside a fish session, so the alias went to ~/.bashrc."""
    shell, reason, _ = detect(home, tmp_path, ["fish"], login="/bin/bash")
    assert shell == "fish" and "login shell is bash" in reason


def test_wrappers_between_the_script_and_the_shell_are_skipped(home, tmp_path):
    assert detect(home, tmp_path, ["env", "-zsh", "login"], login="/bin/bash")[0] == "zsh"
    assert detect(home, tmp_path, ["/opt/homebrew/bin/fish"])[0] == "fish"
    assert detect(home, tmp_path, ["bash"], login="/usr/bin/fish")[0] == "bash"      # the reverse case


def test_falls_back_to_the_login_shell_then_to_nothing(home, tmp_path):
    shell, reason, _ = detect(home, tmp_path, ["python3", "sudo", "login"], login="/bin/zsh")
    assert shell == "zsh" and "login shell" in reason
    assert detect(home, tmp_path, ["python3"], login="")[0] == ""
    assert detect(home, tmp_path, ["python3"], login="/bin/tcsh")[0] == ""            # unsupported => ask the user


def test_explicit_choice_wins_and_bad_values_are_rejected(home, tmp_path):
    assert detect(home, tmp_path, ["fish"], arg="zsh")[0] == "zsh"
    assert fn(f'{FAKE_PS}\ndetect_shell; echo $DETECTED_SHELL', home, tmp_path, C1="fish", MOTION_SHELL="bash").stdout.strip() == "bash"
    r = install(home, tmp_path, "--shell", "tcsh")
    assert r.returncode == 2 and "Unsupported shell" in r.stderr


@pytest.mark.parametrize("raw,expected", [("-zsh", "zsh"), ("/usr/bin/FISH", "fish"), ("bash", "bash"), ("sh", ""), ("", ""), ("-tcsh", "")])
def test_normalize_shell(home, tmp_path, raw, expected):
    assert fn(f'normalize_shell "{raw}"', home, tmp_path).stdout == expected


def test_real_ancestor_detection_from_a_process_named_fish(home, tmp_path):
    """End to end: run the installer from a process called `fish` while $SHELL says bash."""
    fake = tmp_path / "fish"
    shutil.copy(shutil.which("bash"), fake)
    fake.chmod(0o755)
    r = subprocess.run([str(fake), "-c", f'bash "{INSTALL}" --shell-only; true'], capture_output=True, text=True,
                       env=env_for(home, tmp_path), cwd=str(tmp_path), timeout=60)
    if "Shell:" not in r.stdout:
        pytest.skip(f"copied bash would not run here: {r.stderr[:100]}")
    assert (home / ".config/fish/config.fish").exists() and not (home / ".bashrc").exists()
    assert "Shell: fish" in r.stdout and "login shell is bash" in r.stdout


# ── where the config lives ──────────────────────────────────────────────────

def test_config_file_locations(home, tmp_path):
    rc = lambda shell, **e: fn(f'rc_file_for {shell}', home, tmp_path, **e).stdout
    assert rc("fish") == f"{home}/.config/fish/config.fish"
    assert rc("fish", XDG_CONFIG_HOME="/x/cfg") == "/x/cfg/fish/config.fish"
    assert rc("zsh") == f"{home}/.zshrc" and rc("zsh", ZDOTDIR="/z") == "/z/.zshrc"
    assert fn('uname() { echo Linux; }; rc_file_for bash', home, tmp_path).stdout == f"{home}/.bashrc"
    assert fn('uname() { echo Darwin; }; rc_file_for bash', home, tmp_path).stdout == f"{home}/.bash_profile"


# ── writing the block ───────────────────────────────────────────────────────

@pytest.mark.parametrize("shell,relpath,needle", [
    ("fish", ".config/fish/config.fish", "function motion"),
    ("zsh", ".zshrc", "alias motion="),
])
def test_installs_into_the_chosen_shells_file_only(home, tmp_path, shell, relpath, needle):
    r = install(home, tmp_path, "--shell", shell)
    assert r.returncode == 0, r.stderr
    text = (home / relpath).read_text()
    assert needle in text and str(tmp_path / "bin" / "motion") in text and count(home / relpath) == 1 and count(home / relpath, END) == 1
    assert not (home / ".bashrc").exists() and not (home / ".bash_profile").exists()   # nothing leaked into bash's files
    launcher = tmp_path / "bin" / "motion"
    assert launcher.exists() and launcher.stat().st_mode & stat.S_IXUSR
    assert f'REPO_DIR="{REPO}"' in launcher.read_text()
    assert "Installing dependencies" not in r.stdout                       # --shell-only really is shell-only


def test_rerunning_never_duplicates_the_block(home, tmp_path):
    for _ in range(3):
        assert install(home, tmp_path, "--shell", "fish").returncode == 0
    f = home / ".config/fish/config.fish"
    assert count(f) == 1 and count(f, END) == 1 and f.read_text().count("function motion") == 1


def test_cleans_up_the_duplicate_blocks_an_old_broken_installer_left(home, tmp_path):
    """Bug 2: three duplicate blocks had piled up in ~/.bashrc because `sed -i ''` (BSD syntax) failed on GNU sed."""
    rc = home / BASH_RC
    block = f'\n{START}\nalias motion="/old/path/motion"\n{END}\n'
    rc.write_text("export A=1\nalias ll='ls -l'\n" + block * 3 + "export B=2\n")
    r = install(home, tmp_path, "--shell", "bash")
    assert r.returncode == 0, r.stderr
    text = rc.read_text()
    assert count(rc) == 1 and "/old/path" not in text and str(tmp_path / "bin" / "motion") in text
    assert text.startswith("export A=1\nalias ll='ls -l'\nexport B=2\n")   # the user's own lines untouched, in order


def test_does_not_depend_on_sed_dash_i(home, tmp_path):
    assert "sed -i" not in INSTALL.read_text()
    # ...and works when `sed -i` is unusable, as with an incompatible sed.
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    real = shutil.which("sed")
    (fakebin / "sed").write_text(f'#!/bin/bash\ncase "$1" in -i*) echo "sed: -i unsupported here" >&2; exit 1;; esac\nexec {real} "$@"\n')
    (fakebin / "sed").chmod(0o755)
    rc = home / BASH_RC
    rc.write_text(f"before\n\n{START}\nalias motion=STALE1\n{END}\n\n{START}\nalias motion=STALE2\n{END}\nafter\n")
    r = install(home, tmp_path, "--shell", "bash", PATH=f"{fakebin}:{os.environ['PATH']}")
    assert r.returncode == 0, r.stderr
    assert count(rc) == 1 and rc.read_text().startswith("before\n") and "STALE" not in rc.read_text()


@pytest.mark.parametrize("content", [
    f"keep me\n{START}\nalias motion=x\nMORE USER CONFIG\n",                    # start, never closed
    f"keep me\n{END}\nalias motion=x\n{START}\nMORE USER CONFIG\n",             # end before start
    f"a\n{START}\n{START}\nb\n{END}\n",                                          # nested start
])
def test_refuses_to_edit_unbalanced_markers_instead_of_deleting_the_rest(home, tmp_path, content):
    """The old `sed '/start/,/end/d'` deleted everything after a lone start marker."""
    rc = home / BASH_RC
    rc.write_text(content)
    r = install(home, tmp_path, "--shell", "bash")
    assert rc.read_text() == content                                            # not one byte changed
    assert r.returncode != 0 and "unbalanced" in r.stderr


def test_a_symlinked_config_stays_a_symlink(home, tmp_path):
    """Dotfiles setups symlink config.fish into a repo; replacing the link would silently detach it."""
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    target = dotfiles / "config.fish"
    target.write_text("set -x EDITOR vim\n")
    link = home / ".config/fish/config.fish"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    assert install(home, tmp_path, "--shell", "fish").returncode == 0
    assert link.is_symlink() and os.readlink(link) == str(target)
    assert target.read_text().startswith("set -x EDITOR vim\n") and count(target) == 1


def test_a_last_line_without_a_newline_is_not_glued_to_the_block(home, tmp_path):
    rc = home / ".zshrc"
    rc.write_bytes(b"export LAST=1")                                             # no trailing newline
    assert install(home, tmp_path, "--shell", "zsh").returncode == 0
    lines = rc.read_text().splitlines()
    assert lines[0] == "export LAST=1" and lines[2] == START


def test_paths_with_spaces_and_quotes_survive_round_trip(home, tmp_path):
    tricky = "/tmp/it's a \"path\" $HOME `x`/motion"
    assert fn(f'eval "printf %s $(sh_quote {shlex.quote(tricky)})"', home, tmp_path).stdout == tricky
    fish = shutil.which("fish")
    if fish:
        quoted = subprocess.run(["bash", "-c", f"source {INSTALL}; fish_quote {shlex.quote(tricky)}"], capture_output=True, text=True).stdout
        out = subprocess.run([fish, "-c", f"printf %s {quoted}"], capture_output=True, text=True)
        assert out.stdout == tricky


@pytest.mark.parametrize("shell,tool,relpath", [("bash", "bash", BASH_RC), ("zsh", "zsh", ".zshrc"), ("fish", "fish", ".config/fish/config.fish")])
def test_generated_config_is_syntactically_valid(home, tmp_path, shell, tool, relpath):
    exe = shutil.which(tool)
    if not exe:
        pytest.skip(f"{tool} not installed")
    assert install(home, tmp_path, "--shell", shell).returncode == 0
    check = subprocess.run([exe, "-n", str(home / relpath)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


# ── messages ────────────────────────────────────────────────────────────────

def test_the_closing_hint_names_the_right_file_for_the_shell(home, tmp_path):
    """Bug 3: users were told to `source` a bash file from fish, which fish cannot parse."""
    fish = install(home, tmp_path, "--shell", "fish").stdout
    assert f"source {home}/.config/fish/config.fish" in fish and ".bashrc" not in fish
    zsh = install(home, tmp_path, "--shell", "zsh").stdout
    assert f"source {home}/.zshrc" in zsh and "fish" not in zsh.split("Quick start:")[1]
    assert "Skipping shell integration" in install(home, tmp_path, "--no-shell").stdout


def test_unknown_shell_and_options_are_handled(home, tmp_path):
    r = subprocess.run(["bash", str(INSTALL), "--shell-only"], capture_output=True, text=True,
                       env=env_for(home, tmp_path, SHELL="/bin/tcsh"), cwd=str(tmp_path), timeout=30)
    # (only meaningful when no ancestor is a supported shell; either outcome must not crash)
    assert r.returncode == 0
    bad = install(home, tmp_path, "--bogus")
    assert bad.returncode == 2 and "Unknown option" in bad.stderr
    assert subprocess.run(["bash", str(INSTALL), "--help"], capture_output=True, text=True).stdout.startswith("Usage:")


# ── uninstall ───────────────────────────────────────────────────────────────

def test_uninstall_removes_the_block_from_every_shell_and_keeps_the_rest(home, tmp_path):
    fish, bashrc, zshrc = home / ".config/fish/config.fish", home / BASH_RC, home / ".zshrc"
    stray = home / ".bashrc" if BASH_RC != ".bashrc" else None       # e.g. left behind by an older, wrong install
    fish.parent.mkdir(parents=True)
    files = {fish: "fish", bashrc: "bash", zshrc: "zsh"}
    for f, name in files.items():
        f.write_text(f"# my {name} config\nexport X=1\n")
    for shell in ("fish", "bash", "zsh"):
        assert install(home, tmp_path, "--shell", shell).returncode == 0
    if stray:
        stray.write_text(f"# stray\n\n{START}\nalias motion=dup\n{END}\n")
    else:
        bashrc.write_text(bashrc.read_text() + f"\n{START}\nalias motion=dup\n{END}\n")   # a duplicate block
    r = subprocess.run(["bash", str(INSTALL), "--uninstall"], capture_output=True, text=True,
                       env=env_for(home, tmp_path), timeout=30)
    assert r.returncode == 0 and r.stdout.count("Removed the motion block") == (4 if stray else 3)
    for f, name in files.items():
        assert f.read_text() == f"# my {name} config\nexport X=1\n"                     # exactly as the user left it
    if stray:
        assert stray.read_text() == "# stray\n"
    again = subprocess.run(["bash", str(INSTALL), "--uninstall"], capture_output=True, text=True,
                           env=env_for(home, tmp_path), timeout=30)
    assert "Nothing to remove" in again.stdout
