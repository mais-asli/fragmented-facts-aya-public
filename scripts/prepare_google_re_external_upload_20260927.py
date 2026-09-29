"""Prepare TAU transfer only after the external human review has been frozen."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / 'data/curated/aya-google-re-external-reviewed-20260927.freeze.json'
OUT = ROOT / 'scripts/tau_google_re_external_files_20260927.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads(FREEZE.read_text(encoding='utf-8'))
    review = ROOT / 'data/review/google_re_external_20260927/CANDIDATES_32_WITH_MARK_SUGGESTIONS_20260927.csv'
    curated = ROOT / 'data/curated/aya-google-re-external-reviewed-20260927.jsonl'
    if (frozen['cohort'] != 'google_re_external_pre_aya_20260927' or
            frozen['reviewer'] != 'Reviewer A' or
            sha(review) != frozen['ReviewerA_review_csv_sha256'] or
            sha(curated) != frozen['curated_sha256']):
        raise ValueError('External human review missing or changed')
    files = [
        'data/curated/aya-google-re-external-reviewed-20260927.jsonl',
        'data/curated/aya-google-re-external-reviewed-20260927.freeze.json',
        'data/review/google_re_external_20260927/CANDIDATES_32_GOOGLE_RE_20260927.csv',
        'data/review/google_re_external_20260927/CANDIDATES_32_GOOGLE_RE_20260927.manifest.json',
        'data/review/google_re_external_20260927/CANDIDATES_32_WITH_MARK_SUGGESTIONS_20260927.csv',
        'data/review/imported_ReviewerA_20260926/templates.json',
        'configs/study_google_re_external_20260927.json',
        'ff.py', 'slurm/screen.slurm', 'slurm/experiment.slurm',
        'project_plan/feasibility/google_re_external_workflow_tau_20260927.py',
        *[str(path.relative_to(ROOT)).replace('\\', '/')
          for path in sorted((ROOT / 'src/fragmented_facts').glob('*.py'))],
    ]
    if len(set(files)) != len(files) or any(not (ROOT / path).is_file() for path in files):
        raise ValueError('External upload list has a duplicate or missing file')
    if OUT.exists():
        if json.loads(OUT.read_text(encoding='utf-8')) != files:
            raise FileExistsError('Existing external transfer list differs')
        print(json.dumps({'file_count': len(files), 'approved_facts': frozen['approved_count'],
                          'output': str(OUT), 'reused': True}))
        return
    OUT.write_text(json.dumps(files, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'file_count': len(files), 'approved_facts': frozen['approved_count'],
                      'output': str(OUT)}))


if __name__ == '__main__':
    main()
