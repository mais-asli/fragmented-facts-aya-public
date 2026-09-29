"""Prepare a source-backed English E4 development review without approving it."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/review/heldout_batch_v1/facts.csv'
OUT = ROOT / 'data/review/heldout_batch_v1/E4_ENGLISH_DEV_REVIEW_20260927.csv'
MANIFEST = ROOT / 'data/review/heldout_batch_v1/E4_ENGLISH_DEV_REVIEW_20260927.manifest.json'

# Published institutional, archival, or specialist historical sources. These
# are leads for an independent human review, not decisions attributed to ReviewerA.
LEADS = {
    'Q319896-P19-Q1874': ('https://en.jabotinsky.org/zeev-jabotinsky/life-story/childhood-and-youth/', 'source_supports', 'Institute says born in Odessa.'),
    'Q318219-P19-Q220': ('https://www.royal.uk/prince-henry-benedict', 'source_supports', 'Royal Family says born in Rome.'),
    'Q13230-P19-Q71': ('https://www.unige.ch/campus/numeros/112/dossier2/', 'source_supports', 'University of Geneva says born in Geneva.'),
    'Q77253-P19-Q1726': ('', 'source_needed', 'Albert I birth Munich lacks a verified institutional source in this pass.'),
    'Q190368-P19-Q87': ('', 'source_needed', 'Appian of Alexandria does not alone verify birthplace Alexandria.'),
    'Q34624-P20-Q641': ('https://www.encyclopedia.com/people/literature-and-arts/music-history-composers-and-performers-biographies/giovanni-gabrieli', 'source_supports', 'Encyclopedia of World Biography says died in Venice.'),
    'Q451444-P20-Q2256': ('https://www.engineeringhalloffame.org/profile/william-murdoch', 'granularity_check', 'Death was in Handsworth; decide whether Birmingham city label is historically/administratively suitable.'),
    'Q549786-P20-Q84': ('https://tile.loc.gov/storage-services/master/gdc/gdcebookspublic/20/19/45/28/78/2019452878/2019452878.pdf', 'source_supports', 'Library of Congress book says Soane died in his London house.'),
    'Q44197-P20-Q90': ('https://catalogue.bnf.fr/ark:/12148/cb11922216p', 'source_supports', 'French national library says died in Paris; Hebrew candidate is corrupted, so use English only unless corrected.'),
    'Q186153-P20-Q16869': ('', 'source_needed', 'Procopius/Constantinople identity and place need a stronger source.'),
    'Q3695910-P159-Q171866': ('https://www.tees.ac.uk/about/visiting/travel.cfm', 'source_supports', 'University address is Middlesbrough.'),
    'Q239663-P159-Q11194': ('', 'source_needed', 'FK Sarajevo headquarters location still needs an official source.'),
    'Q1130172-P159-Q486479': ('https://www.mayoclinic.org/about-mayo-clinic/contact', 'source_supports', 'Mayo Clinic central Rochester campus/address; verify intended headquarters interpretation.'),
    'Q223041-P159-Q84': ('https://www.itftennis.com/media/2119/itf-beach-tennis-tour-organisational-guidelines-2019.pdf', 'source_supports', 'ITF head office is London.'),
    'Q912887-P159-Q350': ('https://www.cambridge.org/gb/files/5913/5281/4338/How_To_Find_Us_2012.pdf', 'source_supports', 'Cambridge University Press address is Cambridge.'),
    'Q2306052-P740-Q9248': ('https://nobelenergy.com/whoweare/nobel_heritage', 'source_supports', 'Nobel heritage ties Branobel founding to Baku; verify founding location wording.'),
    'Q152433-P740-Q49218': ('https://www.rochester.edu/newscenter/review-sept-oct-2013-chester-carlson-haloid-xerox-electrophotography/', 'source_supports', 'University of Rochester identifies Haloid/Xerox corporate lineage in Rochester.'),
    'Q218255-P740-Q2256': ('https://www.electriclightorchestra.com/bio.htm', 'source_supports', 'Band biography says formed in Birmingham.'),
    'Q910379-P740-Q220': ('', 'source_needed', 'Leonardo/Finmeccanica corporate continuity and Rome founding need checking.'),
    'Q2573130-P740-Q801': ('', 'wrong_granularity', 'Israel is a country, not a city; do not use for city-level E4 donor pool.'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError('E4 dev review already prepared; preserve the first version')
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        all_facts = list(csv.DictReader(stream))
    selected = all_facts[:20]
    if len(selected) != 20 or {r['fact_id'] for r in selected} != set(LEADS):
        raise ValueError('Frozen 20 development candidates differ')
    fields = ['fact_id', 'relation', 'subject_en', 'answer_en', 'source_url',
              'precheck_status', 'precheck_note', 'ReviewerA_decision_keep_or_reject',
              'ReviewerA_verified_source_and_granularity', 'ReviewerA_reviewer', 'ReviewerA_notes']
    with OUT.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            url, status, note = LEADS[row['fact_id']]
            writer.writerow({'fact_id': row['fact_id'], 'relation': row['relation'],
                             'subject_en': row['subject_en'], 'answer_en': row['object_en'],
                             'source_url': url, 'precheck_status': status,
                             'precheck_note': note})
    manifest = {'purpose': 'English-only E4 dev source and granularity review; no Aya E4 outputs inspected',
                'candidate_manifest_sha256': sha(ROOT / 'data/review/heldout_batch_v1/manifest.json'),
                'facts_csv_sha256': sha(SOURCE), 'review_csv_sha256': sha(OUT),
                'count': len(selected), 'reviewer_status': 'pending; all ReviewerA fields blank'}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'path': str(OUT), 'candidates': len(selected),
                      'source_supported_leads': sum(v[1] == 'source_supports' for v in LEADS.values())}))


if __name__ == '__main__':
    main()
