"""Retrieve 2023 English Wikipedia traffic for reviewed Google-RE subjects.

This is an outcome-unseen historical popularity proxy, not Aya training frequency.
The Wikidata sitelink and all 12 monthly API responses are retained and hashed.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote, urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from retrieve_historical_pageviews_20260927 import fetch, sha, atomic_json  # noqa: E402

FACTS = ROOT / 'data/curated/aya-google-re-external-reviewed-20260927.jsonl'
FREEZE = ROOT / 'data/curated/aya-google-re-external-reviewed-20260927.freeze.json'
CACHE = ROOT / 'data/external/wikimedia_2023_google_re'
PROTOCOL = CACHE / 'protocol.json'
OUT = ROOT / 'results/pageviews-2023-google-re-26.csv'
MANIFEST = ROOT / 'results/pageviews-2023-google-re-26.manifest.json'


def main():
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError('External pageview proxy already finalized')
    frozen = json.loads(FREEZE.read_text(encoding='utf-8'))
    facts = [json.loads(line) for line in FACTS.read_text(encoding='utf-8').splitlines()]
    if (frozen['approved_count'] != 26 or len(facts) != 26 or
            frozen['curated_sha256'] != sha(FACTS) or
            len({fact['subject_qid'] for fact in facts}) != 26):
        raise ValueError('External reviewed cohort changed')
    qids = sorted(fact['subject_qid'] for fact in facts)
    definition = {
        'scope': 'Outcome-unseen external-cohort historical popularity proxy; not pretraining frequency',
        'facts_sha256': sha(FACTS), 'review_freeze_sha256': sha(FREEZE),
        'subject_qids': qids,
        'sitelink_endpoint': 'https://www.wikidata.org/w/api.php',
        'views_endpoint': 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/',
        'project': 'en.wikipedia.org', 'access': 'all-access', 'agent': 'user',
        'granularity': 'monthly', 'start': '2023010100', 'end': '2023123100',
        'title_rule': 'current enwiki sitelink for the frozen subject QID; no label-based guess',
        'missing_rule': 'No sitelink, HTTP 404, or fewer than 12 months is missing; no imputation',
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text(encoding='utf-8')) != definition:
            raise ValueError('Existing external pageview protocol differs')
    else:
        atomic_json(PROTOCOL, definition)
    raw_files = [PROTOCOL]
    titles = {}
    for offset in range(0, len(qids), 50):
        batch = qids[offset:offset + 50]
        query = urlencode({'action': 'wbgetentities', 'ids': '|'.join(batch),
                           'props': 'sitelinks', 'sitefilter': 'enwiki', 'format': 'json'})
        url = definition['sitelink_endpoint'] + '?' + query
        path = CACHE / f'sitelinks-batch-{offset // 50:02d}.json'
        raw_files.append(path)
        response = fetch(url, path)
        if response['http_status'] != 200 or response['response'].get('success') != 1:
            raise ValueError('External Wikidata sitelink batch failed')
        entities = response['response']['entities']
        for qid in batch:
            titles[qid] = entities.get(qid, {}).get('sitelinks', {}).get('enwiki', {}).get('title')
    rows = []
    for index, fact in enumerate(sorted(facts, key=lambda f: f['fact_id']), 1):
        qid = fact['subject_qid']
        title = titles[qid]
        record = {'fact_id': fact['fact_id'], 'relation': fact['relation'],
                  'subject_qid': qid, 'enwiki_title': title,
                  'views_2023': None, 'months_observed': 0,
                  'missing_reason': None, 'request_url': None,
                  'raw_response_sha256': None}
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
                items = [item for item in response['response'].get('items', [])
                         if item.get('timestamp', '').startswith('2023')]
                if (len(items) == 12 and len({item['timestamp'][:6] for item in items}) == 12
                        and all(isinstance(item.get('views'), int) and item['views'] >= 0
                                for item in items)):
                    record['views_2023'] = sum(item['views'] for item in items)
                    record['months_observed'] = 12
                else:
                    record['months_observed'] = len(items)
                    record['missing_reason'] = 'incomplete_2023_months'
        else:
            record['missing_reason'] = 'no_enwiki_sitelink'
        rows.append(record)
        if index % 5 == 0:
            print(f'External pageviews: {index}/{len(facts)} subjects', flush=True)
    fields = ['fact_id', 'relation', 'subject_qid', 'enwiki_title', 'views_2023',
              'months_observed', 'missing_reason', 'request_url', 'raw_response_sha256']
    temporary = OUT.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, OUT)
    atomic_json(MANIFEST, {
        'protocol_sha256': sha(PROTOCOL), 'facts_sha256': sha(FACTS),
        'review_freeze_sha256': sha(FREEZE), 'result_csv_sha256': sha(OUT),
        'facts': len(rows), 'complete_2023_subjects': sum(r['views_2023'] is not None for r in rows),
        'missing_by_reason': {reason: sum(r['missing_reason'] == reason for r in rows)
                              for reason in sorted({r['missing_reason'] for r in rows if r['missing_reason']})},
        'raw_response_files': [{'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                                'sha256': sha(path)} for path in raw_files],
        'retrieved_before_external_aya_outcomes': True,
    })
    print(json.dumps({'output': str(OUT), 'complete': sum(r['views_2023'] is not None for r in rows)}))


if __name__ == '__main__':
    main()
