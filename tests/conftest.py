"""Test-suite safety net.

The tests exercise a tool that runs shell commands, including tests about
*refusing* destructive ones. A bug once let `rm -rf ~` run for real during a
test. So the whole suite runs with HOME pointing at a throwaway directory:
`~`, `$HOME`, Path.home() and config/auth lookups all resolve inside it. This
happens at import time, before any test module imports code that reads HOME.
"""
import atexit
import os
import shutil
import tempfile

_FAKE_HOME = tempfile.mkdtemp(prefix="motion-test-home-")
os.environ["HOME"] = _FAKE_HOME
os.environ["USERPROFILE"] = _FAKE_HOME  # Windows
os.environ.pop("MOTION_AUTH_DIR", None)
atexit.register(shutil.rmtree, _FAKE_HOME, ignore_errors=True)
