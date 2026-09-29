"""Freeze external high/low fragmentation matches before seeing Aya outcomes."""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from freeze_e3_position_controls_20260927 import prompt_for_local_tokenizer  # noqa: E402
from match_fragmentation_heldout_exploratory import pilot_thresholds, PILOT_TOKENS  # noqa: E402
from match_pageviews_heldout_exploratory_20260927 import (  # noqa: E402
    match_group, LOG_VIEW_CALIPER, LENGTH_CALIPER,
)
from fragmented_facts.prompts import template_for  # noqa: E402
from fragmented_facts.unicode import base_length  # noqa: E402

FACTS = ROOT / 'data/curated/aya-google-re-external-reviewed-20260927.jsonl'
FREEZE = ROOT / 'data/curated/aya-google-re-external-reviewed-20260927.freeze.json'
PAGEVIEWS = ROOT / 'results/pageviews-2023-google-re-26.csv'
PAGEVIEWS_MANIFEST = ROOT / 'results/pageviews-2023-google-re-26.manifest.json'
TEMPLATES = ROOT / 'data/review/imported_ReviewerA_20260926/templates.json'
TOKENIZER = ROOT / 'data/raw/aya_23_8b/tokenizer.json'
OUT = ROOT / 'results/google-re-pageviews-2023-pre-aya-match-20260927.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('External pageview match already frozen')
    frozen = json.loads(FREEZE.read_text(encoding='utf-8'))
    manifest = json.loads(PAGEVIEWS_MANIFEST.read_text(encoding='utf-8'))
    if (frozen['curated_sha256'] != sha(FACTS) or manifest['result_csv_sha256'] != sha(PAGEVIEWS)
            or manifest['facts_sha256'] != sha(FACTS)):
        raise ValueError('External reviewed facts or pageviews changed')
    facts = [json.loads(line) for line in FACTS.read_text(encoding='utf-8').splitlines()]
    with PAGEVIEWS.open(encoding='utf-8', newline='') as handle:
        pageviews = {row['fact_id']: row for row in csv.DictReader(handle)}
    if len(facts) != 26 or len(pageviews) != 26:
        raise ValueError('Expected 26 reviewed external facts')
    thresholds = pilot_thresholds()
    templates = json.loads(TEMPLATES.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER.parent),
                                              local_files_only=True, use_fast=True)
    records = []
    for fact in facts:
        view = pageviews[fact['fact_id']]
        if fact['subject_qid'] != view['subject_qid']:
            raise ValueError('External pageview subject mismatch')
        for language in ('he', 'ar'):
            name = fact['pairs'][language]['U']
            prompt = prompt_for_local_tokenizer(
                tokenizer, template_for(templates, fact['relation'], language, 't1'),
                name, language)
            length = base_length(name, language)
            density = len(prompt.subject_indices) / length
            threshold = thresholds[language, fact['relation']]
            records.append({
                'fact_id': fact['fact_id'], 'language': language,
                'relation': fact['relation'], 'subject_qid': fact['subject_qid'],
                'views_2023': int(view['views_2023']) if view['views_2023'] else None,
                'missing_reason': view['missing_reason'], 'base_characters': length,
                'u_subject_tokens_t1': len(prompt.subject_indices),
                'u_token_density_t1': density,
                'pilot_relation_median_density': threshold,
                'group': 'high' if density > threshold else 'low',
            })
    groups, pairs = [], []
    for language in ('he', 'ar'):
        for relation in ('P19', 'P20'):
            cohort = [r for r in records if r['language'] == language and r['relation'] == relation]
            observed = [r for r in cohort if r['views_2023'] is not None]
            matches = match_group(observed)
            group_pairs = []
            for high, low in matches:
                pair = {
                    'language': language, 'relation': relation,
                    'high_fact_id': high['fact_id'], 'low_fact_id': low['fact_id'],
                    'high_views_2023': high['views_2023'], 'low_views_2023': low['views_2023'],
                    'abs_log1p_views_gap': abs(math.log1p(high['views_2023']) - math.log1p(low['views_2023'])),
                    'high_base_characters': high['base_characters'],
                    'low_base_characters': low['base_characters'],
                    'abs_base_character_gap': abs(high['base_characters'] - low['base_characters']),
                    'high_token_density': high['u_token_density_t1'],
                    'low_token_density': low['u_token_density_t1'],
                }
                group_pairs.append(pair)
                pairs.append(pair)
            groups.append({'language': language, 'relation': relation,
                           'subjects': len(cohort), 'missing_pageviews': len(cohort) - len(observed),
                           'high_with_views': sum(r['group'] == 'high' for r in observed),
                           'low_with_views': sum(r['group'] == 'low' for r in observed),
                           'matched_pairs': len(group_pairs), 'pairs': group_pairs})
    if any(p['high_token_density'] <= p['low_token_density'] for p in pairs):
        raise ValueError('External high/low match has reversed density')
    payload = {
        'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'Assigned before inspecting any Google-RE Aya outputs',
        'scope': 'Google-RE outcome-unseen observational historical-popularity match',
        'cohort': '26 personally reviewed P19/P20 facts, separate benchmark archive',
        'matching': 'Within relation and language, no replacement, maximum cardinality then minimum distance',
        'threshold': 'Fixed pilot relation/language median of U tokens per base letter at template t1',
        'calipers': {'abs_log1p_2023_pageviews': LOG_VIEW_CALIPER,
                    'abs_base_characters': LENGTH_CALIPER},
        'missing': 'Exclude subjects without all 12 months; no imputation',
        'outcomes_to_join_later': 'Three-template U entity accuracy and gold-answer log likelihood from external E2 only',
        'source_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path)
                          for path in (FACTS, FREEZE, PAGEVIEWS, PAGEVIEWS_MANIFEST,
                                       PILOT_TOKENS, TEMPLATES, TOKENIZER)},
        'subjects': records, 'groups': groups,
        'total_pairs': len(pairs),
        'pairs_by_language': dict(Counter(p['language'] for p in pairs)),
        'limits': ['Pageviews are public attention, not Aya pretraining counts.',
                   'Names and subject identities differ across matched pairs.',
                   'Only P19/P20 exist in the separate Google-RE archive.',
                   'A small number of matches gives imprecise descriptive estimates.'],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'pairs': len(pairs),
                      'by_group': {f"{g['language']}/{g['relation']}": g['matched_pairs'] for g in groups},
                      'sha256': sha(OUT)}))


if __name__ == '__main__':
    main()
