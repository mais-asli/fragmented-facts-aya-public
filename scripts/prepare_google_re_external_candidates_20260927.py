"""Freeze an outcome-unseen external Google-RE candidate list.

The 2019 Meta LAMA Google-RE subset is a different source archive from
mLAMA-1.1. Wikidata is used only to find current Hebrew/Arabic name candidates;
all original Google-RE facts and their evidence are retained for human review.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlencode
import zipfile

ROOT = Path(__file__).resolve().parents[1]
GOOGLE_ZIP = ROOT / 'data/external/lama_google_re/LAMA_data_2019.zip'
MLAMA_ZIP = ROOT / 'data/raw/mlama1.1.zip'
LEDGER = ROOT / 'output/data_audit_20260927/FROZEN_FACT_LEDGER_97.csv'
CACHE = ROOT / 'data/external/lama_google_re'
OUT = ROOT / 'data/review/google_re_external_20260927/CANDIDATES_32_GOOGLE_RE_20260927.csv'
MANIFEST = OUT.with_suffix('.manifest.json')
USER_AGENT = 'AyaFragmentedFactsResearch/1.0 (noncommercial academic project)'
QID = re.compile(r'Q[1-9][0-9]*\Z')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def get_json(url, path):
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['url'] != url:
            raise ValueError('Cached Wikidata URL differs')
        return saved['response']
    body = path.with_name(path.name + '.body.tmp')
    result = subprocess.run(['curl.exe', '-sS', '-L', '--max-time', '30',
                             '--retry', '3', '--retry-all-errors', '--retry-delay', '2',
                             '-H', 'User-Agent: ' + USER_AGENT,
                             '-o', str(body), '-w', '%{http_code}', '--url', url],
                            cwd=ROOT, capture_output=True, text=True, timeout=150)
    raw = body.read_bytes() if body.exists() else b''
    if body.exists():
        body.unlink()
    if result.returncode or result.stdout.strip() != '200':
        raise RuntimeError('Wikidata API response failed: ' + result.stdout + ' ' + result.stderr[-200:])
    parsed = json.loads(raw)
    if parsed.get('success') != 1:
        raise ValueError('Wikidata API returned no success')
    atomic_json(path, {'url': url, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                       'response_sha256': hashlib.sha256(raw).hexdigest(), 'response': parsed})
    time.sleep(0.35)
    return parsed


def existing_mlama_subjects():
    subjects = set()
    with zipfile.ZipFile(MLAMA_ZIP) as archive:
        for language in ('en', 'he', 'ar'):
            for relation in ('P19', 'P20', 'P159', 'P740'):
                for line in archive.open(f'mlama1.1/{language}/{relation}.jsonl'):
                    subjects.add(json.loads(line)['sub_uri'])
    return subjects


def google_pool(excluded, approved_answers):
    pool = []
    with zipfile.ZipFile(GOOGLE_ZIP) as archive:
        for name, relation in (('place_of_birth', 'P19'), ('place_of_death', 'P20')):
            for line_number, line in enumerate(archive.open(f'data/Google_RE/{name}_test.jsonl'), 1):
                item = json.loads(line)
                subject, answer = str(item.get('sub_w')), str(item.get('obj_w'))
                if (not QID.fullmatch(subject) or not QID.fullmatch(answer)
                        or subject in excluded or answer not in approved_answers):
                    continue
                if sum(j.get('judgment') == 'yes' for j in item.get('judgments', [])) < 3:
                    continue
                evidences = [e for e in item.get('evidences', [])
                             if item['obj_label'].casefold() in e.get('snippet', '').casefold()
                             and e.get('url', '').startswith('http')]
                if not evidences:
                    continue
                pool.append({'relation': relation, 'source_path': f'Google_RE/{name}_test.jsonl',
                             'source_line': line_number, 'source_uuid': item['uuid'],
                             'subject_qid': subject, 'answer_qid': answer,
                             'subject_en_google_re': item['sub_label'],
                             'answer_en_google_re': item['obj_label'],
                             'evidence_url': evidences[0]['url'],
                             'evidence_snippet': evidences[0]['snippet'],
                             'judgments_yes': sum(j.get('judgment') == 'yes' for j in item['judgments']),
                             'judgments_total': len(item['judgments'])})
    unique = {}
    for row in pool:
        key = (row['subject_qid'], row['relation'])
        if key not in unique:
            unique[key] = row
    by_relation = {}
    for relation in ('P19', 'P20'):
        entries = [r for r in unique.values() if r['relation'] == relation]
        by_relation[relation] = sorted(entries, key=lambda r: hashlib.sha256(
            ('17|google-re-external|' + r['source_uuid']).encode()).hexdigest())
    return by_relation


def main():
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError('External candidate list already frozen')
    with LEDGER.open(encoding='utf-8-sig', newline='') as source:
        facts = list(csv.DictReader(source))
    answer_labels = {}
    for fact in facts:
        answer_labels.setdefault(fact['answer_qid'], {
            'en': fact['answer_en'], 'he': fact['answer_he'], 'ar': fact['answer_ar']})
    excluded = existing_mlama_subjects() | {f['subject_qid'] for f in facts}
    pools = google_pool(excluded, set(answer_labels))
    # Query every eligible source row. Any language-label exclusions therefore
    # happen before seeing Aya outcomes, without an arbitrary top-100 cap.
    shortlist = pools
    qids = sorted({row['subject_qid'] for rows in shortlist.values() for row in rows})
    labels = {}
    raw_paths = []
    for offset in range(0, len(qids), 50):
        batch = qids[offset:offset + 50]
        query = urlencode({'action': 'wbgetentities', 'ids': '|'.join(batch),
                           'props': 'labels', 'languages': 'en|he|ar', 'format': 'json'})
        url = 'https://www.wikidata.org/w/api.php?' + query
        path = CACHE / f'google-re-wikidata-labels-all-{offset // 50:02d}.json'
        raw_paths.append(path)
        response = get_json(url, path)
        for qid in batch:
            labels[qid] = {lang: response['entities'].get(qid, {}).get('labels', {}).get(lang, {}).get('value')
                           for lang in ('en', 'he', 'ar')}
        print(f'Loaded labels for {min(offset + 50, len(qids))}/{len(qids)} subjects', flush=True)
    selected = []
    for relation in ('P19', 'P20'):
        options = []
        for row in shortlist[relation]:
            langs = labels.get(row['subject_qid'], {})
            if (not langs.get('he') or not langs.get('ar') or
                    not re.search('[\u0590-\u05ff]', langs['he']) or
                    not re.search('[\u0600-\u06ff]', langs['ar'])):
                continue
            options.append({**row, 'subject_en_wikidata': langs['en'],
                            'subject_he_U_candidate': langs['he'],
                            'subject_ar_U_candidate': langs['ar'],
                            'answer_en_reviewed': answer_labels[row['answer_qid']]['en'],
                            'answer_he_reviewed': answer_labels[row['answer_qid']]['he'],
                            'answer_ar_reviewed': answer_labels[row['answer_qid']]['ar']})
        print(f'{relation}: {len(options)} bilingual names in the external source pool', flush=True)
        if len(options) < 16:
            raise ValueError(f'Fewer than 16 bilingual candidate names for {relation}: {len(options)}')
        selected += options[:16]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ['relation', 'subject_qid', 'answer_qid', 'subject_en_google_re',
              'subject_en_wikidata', 'answer_en_google_re', 'answer_en_reviewed',
              'subject_he_U_candidate', 'subject_ar_U_candidate',
              'answer_he_reviewed', 'answer_ar_reviewed',
              'evidence_url', 'evidence_snippet', 'judgments_yes', 'judgments_total',
              'source_path', 'source_line', 'source_uuid',
              'subject_he_D_proposed', 'subject_ar_D_proposed',
              'ReviewerA_fact_and_evidence_keep_or_reject',
              'ReviewerA_he_U_D_approved', 'ReviewerA_ar_U_D_approved',
              'ReviewerA_reviewer', 'ReviewerA_notes']
    with OUT.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    report = {'purpose': 'External Google-RE archive, birth/death place only; pre-Aya candidates for human review',
              'google_archive_sha256': sha(GOOGLE_ZIP), 'mlama_archive_sha256': sha(MLAMA_ZIP),
              'frozen_97_ledger_sha256': sha(LEDGER),
              'exclusion': 'No subject in mLAMA en/he/ar P19/P20/P159/P740 partitions; answer QID must be in ReviewerA-reviewed 97-fact ledger',
              'selection': 'SHA256(17|google-re-external|source_uuid) over all eligible source rows; require Hebrew/Arabic Wikidata labels; first 16 each',
              'google_relation_count_after_filter': {relation: len(rows) for relation, rows in pools.items()},
              'candidates': len(selected), 'by_relation': {'P19': 16, 'P20': 16},
              'review_status': 'all ReviewerA columns blank; no Aya outcomes read',
              'candidate_csv_sha256': sha(OUT),
              'wikidata_label_responses': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in raw_paths]}
    atomic_json(MANIFEST, report)
    print(json.dumps({'path': str(OUT), 'candidates': len(selected),
                      'pool': report['google_relation_count_after_filter']}))


if __name__ == '__main__':
    main()
