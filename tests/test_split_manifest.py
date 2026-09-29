import copy
import pytest
from fragmented_facts.data import (apply_split_manifest, create_split_manifest,
                                   csv_rows, prepare, review_export, write_csv)
from fragmented_facts.io import read_json, read_jsonl, write_json, write_jsonl


def test_approval_growth_keeps_existing_subject_splits(tmp_path, facts):
    source = tmp_path / 'candidates.jsonl'
    rows = copy.deepcopy(facts)
    for row in rows:
        row['wikidata'] = {'single_answer_agrees': True}
    write_jsonl(source, rows)
    manifest = tmp_path / 'splits.json'
    create_split_manifest(source, manifest)
    review_export(source, tmp_path / 'review')
    review_path = tmp_path / 'review/facts.csv'
    reviews = csv_rows(review_path)
    for row in reviews:
        row.update(decision='approved', reviewer='software-test-fixture',
                   pre_release_verified='true', evidence_url='https://example.org/fixture')
    output = tmp_path / 'accepted.jsonl'
    write_csv(review_path, reviews[:5], list(reviews[0]))
    prepare(source, review_path, tmp_path / 'review/pairs.csv', output, split_manifest=manifest)
    first = {f['subject_qid']: f['split'] for f in read_jsonl(output)}
    write_csv(review_path, reviews, list(reviews[0]))
    prepare(source, review_path, tmp_path / 'review/pairs.csv', output, split_manifest=manifest)
    second = {f['subject_qid']: f['split'] for f in read_jsonl(output)}
    assert all(second[q] == split for q, split in first.items())


def test_pinned_pool_and_seed_cannot_change_silently(tmp_path, facts):
    source, manifest = tmp_path / 'facts.jsonl', tmp_path / 'splits.json'
    write_jsonl(source, facts)
    create_split_manifest(source, manifest)
    with pytest.raises(FileExistsError):
        create_split_manifest(source, manifest)
    with pytest.raises(ValueError, match='seed'):
        apply_split_manifest(source, manifest, seed=99)
    write_jsonl(source, facts[:-1])
    with pytest.raises(ValueError, match='pool changed'):
        apply_split_manifest(source, manifest)


def test_edited_assignment_rejected(tmp_path, facts):
    source, manifest = tmp_path / 'facts.jsonl', tmp_path / 'splits.json'
    write_jsonl(source, facts)
    create_split_manifest(source, manifest)
    edited = read_json(manifest)
    edited['subjects'][facts[0]['subject_qid']] = 'test'
    edited['seed'] = 42
    write_json(manifest, edited)
    with pytest.raises(ValueError, match='changed'):
        apply_split_manifest(source, manifest)
