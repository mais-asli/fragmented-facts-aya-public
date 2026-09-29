"""Validate and export the completed TAU smoke evidence already returned over SSH."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source_path = ROOT / 'results/tau_workflow.json'
source_bytes = source_path.read_bytes()
workflow = json.loads(source_bytes)
upload = json.loads((ROOT / 'results/tau_upload.json').read_text(encoding='utf-8'))
local_snapshot = json.loads((ROOT / 'model_snapshot.json').read_text(encoding='utf-8'))
assert upload['status'] == 'uploaded_and_sha256_verified'
assert upload['completed_files'] == upload['total_files']
assert upload['verified_bytes'] == upload['total_bytes']
assert workflow['status'] == 'workflow_finished' and workflow['workflow_exit_code'] == 0
remote = workflow['workflow']
smoke = remote['smoke_report']
assert smoke['status'] == 'mechanical_checks_passed' and smoke['checks']
assert all(item['passed'] is True for item in smoke['checks'])
assert {item['language'] for item in smoke['examples']} == {'en', 'he', 'ar'}
assert all(item['answer'].strip() for item in smoke['examples'])
assert smoke['model_source']['model_id'] == local_snapshot['model_id'] == 'CohereLabs/aya-23-8B'
assert smoke['model_source']['revision'] == local_snapshot['revision']
assert smoke['model_source']['files'] == local_snapshot['files']
assert smoke['precision'] == 'nf4' and smoke['compute_dtype'] == 'fp16'
job_id = remote['job_id']
assert job_id.isdigit()
accounting_rows = [line.split('|') for line in remote['accounting'].splitlines()]
job_row = next(row for row in accounting_rows if row[0] == job_id)
assert job_row[1] == 'COMPLETED' and job_row[-1] == '0:0'
report_name = f'aya-smoke-nf4-{job_id}.json'
(ROOT / 'results' / report_name).write_text(
    json.dumps(smoke, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
summary = {
    'status': 'actual_aya_gpu_smoke_passed',
    'job_id': job_id, 'job_state': job_row[1], 'elapsed': job_row[2],
    'finished_at': workflow['finished_at'],
    'source_report': 'results/tau_workflow.json',
    'source_report_sha256': hashlib.sha256(source_bytes).hexdigest(),
    'smoke_report': 'results/' + report_name,
    'model_id': smoke['model_source']['model_id'],
    'revision': smoke['model_source']['revision'],
    'gpu': smoke['gpu'], 'precision': smoke['precision'],
    'compute_dtype': smoke['compute_dtype'],
    'checks_passed': len(smoke['checks']), 'examples': len(smoke['examples']),
    'peak_allocated_gib': smoke['peak_allocated_gib'],
    'peak_reserved_gib': smoke['peak_reserved_gib'],
    'model_load_seconds': smoke['load_seconds'],
    'packages': smoke['packages'],
    'research_experiments_completed': False,
    'remaining': smoke['next_checks'],
}
(ROOT / 'results/aya-feasibility-summary.json').write_text(
    json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))

production_path = ROOT / 'results/tau_production_workflow.json'
if production_path.is_file():
    production_bytes = production_path.read_bytes()
    production = json.loads(production_bytes)
    assert production['status'] == 'workflow_finished' and production['workflow_exit_code'] == 0
    remote = production['workflow']
    report = remote['model_report']
    assert report['status'] == 'production_checks_passed'
    assert report['checks'] and all(c['passed'] for c in report['checks'])
    assert report['model']['revision'] == local_snapshot['revision']
    assert report['data_kind'] == 'synthetic_fixture'
    job_id = remote['job_id']
    terminal = next(row.split('|') for row in remote['accounting'].splitlines() if row.split('|')[0] == job_id)
    assert terminal[1] == 'COMPLETED' and terminal[-1] == '0:0'
    for item in report['fixture_runs']:
        assert item['first']['total_items'] == item['first']['expected_count']
        assert item['resume']['completed_now'] == 0
        assert item['resume']['resumed_items'] == item['first']['total_items']
    (ROOT / f'results/aya-production-check-{job_id}.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'requirements-cluster-validated.txt').write_text(remote['installed_packages'], encoding='utf-8')
    production_summary = {
        'status': report['status'], 'job_id': job_id, 'elapsed': terminal[2],
        'checks_passed': len(report['checks']), 'fixture_items':sum(x['first']['total_items'] for x in report['fixture_runs']),
        'source_code_hash': report['source_code_hash'],
        'workflow_sha256': hashlib.sha256(production_bytes).hexdigest(),
        'model': report['model'], 'runtime': report['runtime'],
        'research_results': False, 'native_review': 'pending',
        'resume_note': 'E1/E2 perform zero resumed forwards; E5 preserves outputs but recomputes donor activations.'}
    (ROOT / 'results/aya-production-summary.json').write_text(json.dumps(production_summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(production_summary, indent=2))
