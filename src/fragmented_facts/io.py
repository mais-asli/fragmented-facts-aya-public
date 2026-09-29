"""Canonical hashes, crash-safe files, and immutable per-item experiment records."""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for i, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{i}") from exc
    return rows


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_text(path, "".join(canonical(row) + "\n" for row in rows))


def code_hash():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


class ResultStore:
    """One directory per run; a process lock prevents concurrent same-run writers.

    A killed process may leave .lock: resume only after its Slurm job has ended.
    Item files are atomic, so no half-written JSONL line can corrupt a run.
    """
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.manifest = manifest
        self.run_id = digest(manifest)
        self.path = self.root / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)
        existing = self.path / "manifest.json"
        if existing.exists() and read_json(existing) != manifest:
            raise ValueError("Run manifest conflict")
        if not existing.exists():
            write_json(existing, manifest)
        self.items = self.path / "items"
        self.items.mkdir(exist_ok=True)
        self.lock = self.path / ".lock"

    def __enter__(self):
        try:
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"Run locked: {self.lock}. Check the previous job before removing this lock.") from exc
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "started": now(), "slurm_job": os.getenv("SLURM_JOB_ID")}, stream)
        return self

    def __exit__(self, *_):
        self.lock.unlink(missing_ok=True)

    def get(self, key):
        path = self.items / (digest(key) + ".json")
        if not path.exists():
            return None
        row = read_json(path)
        if row.get("key") != key or row.get("status") != "complete":
            raise ValueError(f"Invalid existing record: {path}")
        return row

    def put(self, key, payload):
        if self.get(key) is not None:
            raise ValueError("Duplicate result key; resume should skip existing items")
        row = {"key": key, "run_id": self.run_id, "status": "complete", "completed_at": now(), **payload}
        write_json(self.items / (digest(key) + ".json"), row)
        return row

    def rows(self):
        return [read_json(p) for p in sorted(self.items.glob("*.json"))]
