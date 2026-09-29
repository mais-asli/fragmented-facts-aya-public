"""Copy the verified Aya snapshot into a dedicated Linux Docker volume."""
import hashlib
import json
import os
from pathlib import Path
import time

source, target = Path('/source'), Path('/models/aya')
manifest = json.loads(Path('/app/results/aya_cpu_snapshot.json').read_text(encoding='utf-8'))
expected = {row['file']: row['sha256'] for row in manifest['files']}
target.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
copied = []
for path in sorted(source.iterdir()):
    if not path.is_file() or path.suffix not in {'.json', '.safetensors', '.model', '.txt', '.md'}:
        continue
    destination = target / path.name
    if destination.exists():
        raise FileExistsError('Destination already exists: ' + path.name)
    temporary = target / (path.name + '.copying')
    checksum = hashlib.sha256()
    with path.open('rb') as reader, temporary.open('xb') as writer:
        for block in iter(lambda: reader.read(16 * 1024 * 1024), b''):
            writer.write(block)
            checksum.update(block)
        writer.flush()
        os.fsync(writer.fileno())
    digest = checksum.hexdigest()
    if path.name in expected and digest != expected[path.name]:
        raise ValueError('Model checksum failed: ' + path.name)
    temporary.replace(destination)
    copied.append({'file': path.name, 'bytes': destination.stat().st_size, 'sha256': digest})
    print('Cached ' + path.name, flush=True)
report = {'status': 'copied_and_verified', 'snapshot_revision': manifest['revision'],
          'volume': 'fragmented-facts-aya-89da1a0', 'files': copied,
          'seconds': time.perf_counter() - started}
Path('/app/results/aya_linux_cache.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Aya is cached in Linux storage and ready to load.', flush=True)
