"""Fetch and preserve 2023 human pageviews for frozen subject QIDs.

The Wikidata enwiki sitelink chooses the article; monthly Wikimedia Analytics
API data measure article traffic, not Aya training frequency. Missing titles or
API observations are never imputed. Reruns reuse recorded raw responses.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import quote, urlencode

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'output/data_audit_20260927/FROZEN_FACT_LEDGER_97.csv'
CACHE = ROOT / 'data/external/wikimedia_2023'
PROTOCOL = CACHE / 'protocol.json'
OUT_CSV = ROOT / 'results/pageviews-2023-frozen-97.csv'
OUT_JSON = ROOT / 'results/pageviews-2023-frozen-97.manifest.json'
USER_AGENT = 'AyaFragmentedFactsResearch/1.0 (noncommercial academic project)'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def fetch(url, path):
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('request_url') != url:
            raise ValueError('Cached URL differs: ' + str(path))
        return payload
    body = path.with_name(path.name + '.body.tmp')
    call = subprocess.run([
        'curl.exe', '-sS', '-L', '--max-time', '25', '--retry', '3',
        '--retry-delay', '2', '--retry-all-errors',
        '-H', 'User-Agent: ' + USER_AGENT, '-H', 'Accept: application/json',
        '-o', str(body), '-w', '%{http_code}', '--url', url,
    ], cwd=ROOT, capture_output=True, text=True, timeout=125)
    status = int(call.stdout.strip() or '0')
    raw = body.read_bytes() if body.exists() else b''
    if body.exists():
        body.unlink()
    if call.returncode or status not in (200, 404):
        raise RuntimeError(f'API request failed: HTTP {status}, curl {call.returncode}, {call.stderr[-300:]}')
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f'API returned non-JSON HTTP {status}') from error
    payload = {'request_url': url, 'http_status': status,
               'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
               'response_sha256': hashlib.sha256(raw).hexdigest(), 'response': parsed}
    atomic_json(path, payload)
    time.sleep(0.35)
    return payload


def main():
    if OUT_CSV.exists() or OUT_JSON.exists():
        raise FileExistsError('Pageview result already finalized; preserve the first retrieval')
    with LEDGER.open(encoding='utf-8-sig', newline='') as source:
        facts = list(csv.DictReader(source))
    if len(facts) != 97:
        raise ValueError('Expected frozen 97-fact ledger')
    qids = sorted({row['subject_qid'] for row in facts})
    if len(qids) != 97:
        raise ValueError('Expected 97 distinct subjects')
    CACHE.mkdir(parents=True, exist_ok=True)
    definition = {
        'scope': 'Post-outcome observational historical popularity proxy, not pretraining frequency',
        'input_ledger_sha256': sha(LEDGER), 'subject_qids': qids,
        'sitelink_endpoint': 'https://www.wikidata.org/w/api.php',
        'views_endpoint': 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/',
        'project': 'en.wikipedia.org', 'access': 'all-access', 'agent': 'user',
        'granularity': 'monthly', 'start': '2023010100', 'end': '2023123100',
        'title_rule': 'current enwiki sitelink for the frozen subject QID; no title-label guess',
        'missing_rule': 'No sitelink, HTTP 404, or fewer than 12 months means missing; no imputation',
    }
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text(encoding='utf-8')) != definition:
            raise ValueError('Existing frozen retrieval protocol differs')
    else:
        atomic_json(PROTOCOL, definition)
    title_by_qid = {}
    raw_files = [PROTOCOL]
    for i in range(0, len(qids), 50):
        batch = qids[i:i + 50]
        query = urlencode({'action': 'wbgetentities', 'ids': '|'.join(batch),
                           'props': 'sitelinks', 'sitefilter': 'enwiki', 'format': 'json'})
        url = definition['sitelink_endpoint'] + '?' + query
        path = CACHE / f'sitelinks-batch-{i // 50:02d}.json'
        raw_files.append(path)
        response = fetch(url, path)
        if response['http_status'] != 200 or response['response'].get('success') != 1:
            raise ValueError('Wikidata sitelink batch failed')
        entities = response['response']['entities']
        for qid in batch:
            title_by_qid[qid] = entities.get(qid, {}).get('sitelinks', {}).get('enwiki', {}).get('title')
    rows = []
    for index, qid in enumerate(qids, 1):
        title = title_by_qid[qid]
        record = {'subject_qid': qid, 'enwiki_title': title,
                  'views_2023': None, 'months_observed': 0,
                  'missing_reason': None, 'request_url': None, 'raw_response_sha256': None}
        if title:
            slug = quote(title.replace(' ', '_'), safe='')
            url = (definition['views_endpoint'] +
                   f'en.wikipedia.org/all-access/user/{slug}/monthly/2023010100/2023123100')
            path = CACHE / f'pageviews-{qid}.json'
            raw_files.append(path)
            response = fetch(url, path)
            record['request_url'] = url
            record['raw_response_sha256'] = response['response_sha256']
            if response['http_status'] == 404:
                record['missing_reason'] = 'http_404'
            else:
                items = response['response'].get('items', [])
                months = [item for item in items if item.get('timestamp', '').startswith('2023')]
                if (len(months) == 12 and len({item['timestamp'][:6] for item in months}) == 12
                        and all(isinstance(item.get('views'), int) and item['views'] >= 0 for item in months)):
                    record['views_2023'] = sum(item['views'] for item in months)
                    record['months_observed'] = 12
                else:
                    record['months_observed'] = len(months)
                    record['missing_reason'] = 'incomplete_2023_months'
        else:
            record['missing_reason'] = 'no_enwiki_sitelink'
        rows.append(record)
        if index % 10 == 0:
            print(f'Processed {index}/{len(qids)} subjects', flush=True)
    fact_rows = [{**{'split': fact['split'], 'fact_id': fact['fact_id'],
                     'relation': fact['relation']},
                  **next(r for r in rows if r['subject_qid'] == fact['subject_qid'])}
                 for fact in facts]
    fields = ['split', 'fact_id', 'relation', 'subject_qid', 'enwiki_title',
              'views_2023', 'months_observed', 'missing_reason', 'request_url',
              'raw_response_sha256']
    temporary = OUT_CSV.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fact_rows)
    os.replace(temporary, OUT_CSV)
    manifest = {'protocol_sha256': sha(PROTOCOL), 'ledger_sha256': sha(LEDGER),
                'result_csv_sha256': sha(OUT_CSV), 'facts': len(fact_rows),
                'complete_2023_subjects': sum(r['views_2023'] is not None for r in rows),
                'missing_by_reason': {reason: sum(r['missing_reason'] == reason for r in rows)
                                      for reason in sorted({r['missing_reason'] for r in rows if r['missing_reason']})},
                'raw_response_files': [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
                                       for path in raw_files]}
    atomic_json(OUT_JSON, manifest)
    print(json.dumps({'output': str(OUT_CSV), 'complete': manifest['complete_2023_subjects'],
                      'missing': manifest['missing_by_reason']}, indent=2))


if __name__ == '__main__':
    main()
