"""Cache the public primary papers needed by the two partner work packages."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data' / 'reference_papers'
PAPERS = [
    ('2023.emnlp-main.751', 'Geva et al. — factual information flow'),
    ('2025.findings-acl.827', 'Fierro et al. — multilingual factual recall'),
    ('2026.findings-eacl.22', 'Inoue et al. — Arabic diacritics'),
    ('2025.acl-long.253', 'Wang et al. — cross-lingual factual inconsistency'),
    ('2021.eacl-main.284', 'Kassner et al. — mLAMA source dataset'),
]


def fetch(item):
    key, title = item
    url = f'https://aclanthology.org/{key}.pdf'
    path = CACHE / f'{key}.pdf'
    if path.exists():
        data = path.read_bytes()
        action = 'existing_cache_verified'
    else:
        request = urllib.request.Request(url, headers={'User-Agent': 'FragmentedFacts-course-reading-pack/1.0'})
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read()
        if not data.startswith(b'%PDF-') or b'%%EOF' not in data[-2048:]:
            raise ValueError(f'Not a complete PDF: {url}')
        with path.open('xb') as target:
            target.write(data)
        action = 'downloaded'
    if not data.startswith(b'%PDF-') or b'%%EOF' not in data[-2048:]:
        raise ValueError(f'Invalid cached PDF: {path}')
    record = {'file': path.name, 'title': title, 'url': url, 'bytes': len(data),
              'sha256': hashlib.sha256(data).hexdigest(), 'action': action}
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


if __name__ == '__main__':
    CACHE.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, PAPERS))
    manifest = {'checked_at': datetime.now(timezone.utc).isoformat(), 'papers': records}
    (CACHE / 'download_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
