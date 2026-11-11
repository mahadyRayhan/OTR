"""
sandbox.py — run short, model-written Python in an isolated Docker container.

For checks, not jobs: counting sites in a given sequence, testing a suggested function, a small
statistic. SciWeaver still only suggests long or heavy code (training, alignments of real data); it is
never run here.

Each run gets a fresh container from the sciweaver-sandbox image (docker/sandbox/Dockerfile: Python 3.12
with numpy, scipy, pandas, scikit-image, scikit-learn, pysam, biopython):

  no network            --network none (database lookups go through lookups.py, on the host)
  read-only system      only /tmp is writable (a 256 MB tmpfs); the code and its files are copied there
  limits                1 CPU, 1 GB memory, 256 processes, a wall-clock timeout (default 30 s)

Without Docker, run_python raises SandboxUnavailable; code is never run on the host instead.

    res = run_python("print(len('ACGT'))")
    res.ok, res.stdout, res.stderr, res.timed_out, res.seconds
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from orchestrator import trace

IMAGE = "sciweaver-sandbox:1"
DOCKERFILE_DIR = Path(__file__).resolve().parents[1] / "docker" / "sandbox"
MAX_OUTPUT = 20_000  # characters kept of stdout and of stderr

_available: bool | None = None


class SandboxUnavailable(RuntimeError):
    pass


@dataclass
class SandboxResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def available() -> bool:
    """Docker is running and the image exists (checked once per process)."""
    global _available
    if _available is None:
        try:
            p = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=20)
            _available = p.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _available = False
    return _available


def build_image() -> None:
    subprocess.run(["docker", "build", "-t", IMAGE, str(DOCKERFILE_DIR)], check=True)


def run_python(code: str, *, files: dict[str, str] | None = None, stdin: str = "", timeout: int = 30,
               memory: str = "1g") -> SandboxResult:
    """Run code as main.py in a fresh container; files are placed next to it."""
    if not available():
        raise SandboxUnavailable(f"Docker image {IMAGE} not available (start Docker; build with "
                                 f"docker build -t {IMAGE} {DOCKERFILE_DIR})")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        for name, text in {**(files or {}), "main.py": code}.items():
            Path(d, name).write_text(text)
        cmd = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only",
               "--tmpfs", "/tmp:rw,size=256m", "--memory", memory, "--cpus", "1", "--pids-limit", "256",
               "-v", f"{d}:/w:ro", IMAGE, "sh", "-c",
               f"cp -r /w /tmp/work && cd /tmp/work && exec timeout -s KILL {int(timeout)} python main.py"]
        try:
            p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout + 60)
            # timeout(1) kills with SIGKILL: exit status 137
            res = SandboxResult(p.returncode, p.stdout[-MAX_OUTPUT:], p.stderr[-MAX_OUTPUT:],
                                p.returncode == 137, round(time.time() - t0, 2))
        except subprocess.TimeoutExpired as e:
            res = SandboxResult(None, (e.stdout or "")[-MAX_OUTPUT:] if isinstance(e.stdout, str) else "",
                                "", True, round(time.time() - t0, 2))
    trace.record_tool_call("sandbox.python", {"code_chars": len(code), "files": sorted(files or {})},
                           {"returncode": res.returncode, "timed_out": res.timed_out, "seconds": res.seconds,
                            "stdout_tail": res.stdout[-300:], "stderr_tail": res.stderr[-300:]})
    return res


def as_dict(res: SandboxResult) -> dict:
    return dict(asdict(res), ok=res.ok)
