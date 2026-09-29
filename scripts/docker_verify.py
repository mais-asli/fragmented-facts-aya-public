"""Run Linux checks and save machine-readable evidence; never run Aya implicitly."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path("/app/results/docker")
root.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
lock = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, text=True, capture_output=True)
(root / "requirements-linux-tested.txt").write_text(lock.stdout, encoding="utf-8")
tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "--basetemp", "/tmp/ff-pytest",
                        "--junitxml", str(root / "tests.xml")], text=True, capture_output=True)
(root / "tests.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")
if tests.returncode:
    print(tests.stdout + tests.stderr)
    raise SystemExit(tests.returncode)
run = subprocess.run([sys.executable, "ff.py", "integration-test", "--output", str(root / "integration")],
                     text=True, capture_output=True)
(root / "integration.log").write_text(run.stdout + run.stderr, encoding="utf-8")
if run.returncode:
    print(run.stdout + run.stderr)
    raise SystemExit(run.returncode)
report = {"status": "passed", "seconds": time.perf_counter() - started,
          "python": sys.version, "platform": sys.platform, "network": "disabled_by_docker_run",
          "test_log": "tests.log", "integration": "integration/verification.json",
          "scope": "CPU software verification with tiny random Cohere; no Aya weights loaded"}
(root / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(tests.stdout)
print(json.dumps(report, indent=2))
