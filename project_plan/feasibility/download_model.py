"""Resolve and download Aya once. Requires the user's own approved HF access."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', choices=['CohereLabs/aya-23-8B'], default='CohereLabs/aya-23-8B')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--output', default='model_snapshot.json')
    parser.add_argument('--cache-dir', default=None, help='Optional model cache directory')
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(args.model_id, revision=args.revision)
    snapshot = snapshot_download(
        repo_id=args.model_id, revision=info.sha,
        cache_dir=args.cache_dir, max_workers=2,
        allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', 'README.md'],
    )
    record = {
        'model_id': args.model_id, 'revision': info.sha,
        'snapshot_path': str(Path(snapshot).resolve()),
        'downloaded_at': datetime.now(timezone.utc).isoformat(),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
