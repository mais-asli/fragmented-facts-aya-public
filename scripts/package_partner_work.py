"""Create self-contained, language-specific work folders and verified ZIPs.

Existing project files and workbooks are copied, never edited. Only package
copies of Markdown receive portable links and package-specific instructions.
"""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAMP = '20260920'
BASE = ROOT / 'outputs' / 'partner_packages' / '2026-09-20'
PDFS = [
    'NLP_course_2025b___project_guidelines.pdf',
    'NLP_Project_Proposal (1).pdf',
    '1711.05240v5.pdf',
    '2020.emnlp-main.248.pdf',
    'CognitiveSkillsInNLP.pdf',
    'SelfConsistency and beyond in the search of decoding method for mitigating hallucinations.pdf',
]
COMMON = [
    'docs/ANNOTATION_GUIDE.md', 'docs/STATUS.md',
    'docs/IMPLEMENTATION_NOTES.md', 'paper/main.tex', 'paper/references.bib',
    'configs/templates.json', 'configs/study.json',
    'data/review/subject_splits.json',
    'data/review/pilot_batch_v1/manifest.json',
    'data/review/pilot_batch_v1/facts.csv',
    'data/review/pilot_batch_v1/pairs.csv',
    'data/review/pilot_batch_v1/workbook_input.json',
    'outputs/01a09b51/pilot_review/workbook_schema.json',
    'data/raw/candidates_final.jsonl',
    'data/raw/candidates_final.enrichment.json',
    'data/raw/mlama1.1.zip.source.json',
    'results/aya-smoke-nf4-890456.json',
    'results/aya-production-check-892882.json',
    'results/aya-feasibility-summary.json',
    'results/aya-production-summary.json',
]
LANGUAGES = [('Hebrew', 'he', 'עברית', 'P19/P20'), ('Arabic', 'ar', 'ערבית', 'P159/P740')]
SOURCES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_bytes(relative):
    path = ROOT / relative
    data = path.read_bytes()
    value = hashlib.sha256(data).hexdigest()
    if relative in SOURCES and SOURCES[relative] != value:
        raise RuntimeError(f'Source changed while packaging: {relative}')
    SOURCES[relative] = value
    return data


def make_package(parent, label, language, display, relations, packet, papers):
    target = parent / (label + '_Partner')
    target.mkdir()
    provenance = {}

    def write(relative, content, source=None, mode='generated'):
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = content.encode('utf-8') if isinstance(content, str) else content
        with path.open('xb') as handle:
            handle.write(payload)
        provenance[relative] = {'mode': mode}
        if source:
            provenance[relative].update(source_path=source, source_sha256=SOURCES[source])

    def copy(source, destination=None):
        write(destination or source, source_bytes(source), source, 'exact_copy')

    def adapted(source, content):
        write(source, content, source, 'package_adaptation')

    for source in PDFS + COMMON:
        copy(source)
    for path in sorted((ROOT / 'paper/bib').glob('*.bib')):
        copy(path.relative_to(ROOT).as_posix())
    for paper in papers['papers']:
        copy('data/reference_papers/' + paper['file'], 'references/papers/' + paper['file'])
    copy('data/reference_papers/download_manifest.json', 'references/papers/download_manifest.json')

    qids = sorted({f['subject_qid'] for f in packet['facts']} | {f['object_qid'] for f in packet['facts']})
    for qid in qids:
        copy(f'data/raw/wikidata/{qid}.json')

    workbook_name = f'Aya_Pilot_Review_{label}_{STAMP}.xlsx'
    workbook_relative = 'outputs/01a09b51/pilot_review/' + workbook_name
    workbook_source = 'outputs/01a09b51/pilot_review/Aya_Pilot_Review.xlsx'
    copy(workbook_source, workbook_relative)

    guides = {}
    for guide_label, _, _, _ in LANGUAGES:
        relative = f'docs/partner_guides/{guide_label.upper()}_PARTNER_GUIDE.md'
        text = source_bytes(relative).decode('utf-8')
        text = text.replace('outputs/01a09b51/pilot_review/Aya_Pilot_Review.xlsx', workbook_relative)
        text = text.replace(f'Aya_Pilot_Review_{label}_YYYYMMDD.xlsx', workbook_name)
        note = ('\n> עותק לחבילת העבודה מ־20 בספטמבר 2026. התחילו ב־[START_HERE](../../START_HERE.md). '
                'הנתיבים מתייחסים לשורש החבילה אחרי חילוץ ה־ZIP. לוח הזמנים נשאר יעד תכנון; מתחילים כעת לפי סדר המשימות.\n\n')
        text = text.replace('\n\n', note, 1)
        adapted(relative, text)
        guides[guide_label] = text

    pilot_source = 'docs/PILOT_REVIEW.md'
    pilot = source_bytes(pilot_source).decode('utf-8').split('## Import and run')[0]
    pilot = pilot.replace('outputs/01a09b51/pilot_review/Aya_Pilot_Review.xlsx', workbook_relative)
    pilot += ('## Handoff for this partner package\n\n'
              'Return the language-specific workbook and notes to ReviewerA. ReviewerA reconciles the two copies by Fact ID '
              'and field ownership, validates the combined workbook, imports reviews and performs all Aya/TAU runs. '
              'This package adaptation omits the server command section because partners do not execute it.\n')
    adapted(pilot_source, pilot)

    plan_source = 'project_plan/RESEARCH_PLAN.md'
    plan = source_bytes(plan_source).decode('utf-8')
    plan = plan.replace('**Current status:** all six supplied PDFs', '**Historical status when first drafted on September 13:** all six supplied PDFs')
    notice = ('\n> **Package note — September 20:** Aya GPU feasibility and production checks have passed. '
              'The reviewed workload pilot and full study remain pending. Earlier access estimates and initial status '
              'below are historical. The partner guides define the current division of work; ReviewerA owns all runs and paper writing.\n\n')
    plan = plan.replace('\n\n', notice, 1)
    adapted(plan_source, plan)

    feasible_source = 'docs/AYA_FEASIBILITY_RESULT.md'
    feasible = source_bytes(feasible_source).decode('utf-8')
    feasible = feasible.replace('- [Original authenticated workflow report](../results/tau_workflow.json)',
                                '- The original authenticated workflow report remains with ReviewerA; the extracted model reports are included here.')
    feasible = feasible.replace('- [Full transfer and verification receipt](../results/tau_upload.json)',
                                '- Server transfer receipts remain with ReviewerA and are not required for partner review.')
    adapted(feasible_source, feasible)

    source_lines = [
        '# מקורות 70 דוגמאות הפיילוט\n',
        'זו רשימת מועמדים לבדיקה, לא רשימת עובדות שאושרו. Primary הוא הסבב הראשון; Reserve הן חלופות לפי הסדר שנקבע. '
        'יש לפתוח את המקורות שאליהם Wikidata מפנה כדי לאמת עובדה. קובצי המטמון משמרים את המצב בעת האיסוף ואינם תחליף לבדיקת ראיה היסטורית.\n',
        '| עדיפות | Fact ID | נושא באנגלית | תשובה מועמדת | נושא ב־Wikidata | תשובה ב־Wikidata | עותק נושא שמור | עותק תשובה שמור |',
        '|---|---|---|---|---|---|---|---|',
    ]
    priorities = {f['fact_id']: f['priority'] for f in packet['manifest']['facts']}
    for fact in packet['facts']:
        subject, answer = fact['subject_qid'], fact['object_qid']
        esc = lambda v: str(v).replace('|', '\\|').replace('\n', ' ')
        source_lines.append(f"| {priorities[fact['fact_id']]} | {fact['fact_id']} | {esc(fact['subject_en'])} | {esc(fact['object_en'])} | "
                            f'[נושא](https://www.wikidata.org/wiki/{subject}) | [תשובה](https://www.wikidata.org/wiki/{answer}) | '
                            f'[JSON](../data/raw/wikidata/{subject}.json) | [JSON](../data/raw/wikidata/{answer}.json) |')
    write('references/PILOT_SOURCES.md', '\n'.join(source_lines) + '\n')

    primary = {'Hebrew': {'2023.emnlp-main.751.pdf', '2025.findings-acl.827.pdf'},
               'Arabic': {'2026.findings-eacl.22.pdf', '2025.acl-long.253.pdf'}}[label]
    readings = ['# מאמרי הקריאה בחבילה\n',
                'המאמרים המקוריים הורדו מ־ACL Anthology. מתחילים במאמרים המסומנים “קריאה ראשית”; היתר נועדו להשוואה ולהבנת מקור הנתונים. '
                'כל טענה למסירה למייס דורשת עמוד/סעיף/טבלה במאמר, לפי המדריך.\n',
                '| עדיפות | מאמר | עותק מקומי | מקור רשמי |', '|---|---|---|---|']
    for paper in papers['papers']:
        order = 'קריאה ראשית' if paper['file'] in primary else 'רקע משותף'
        readings.append(f"| {order} | {paper['title']} | [PDF](papers/{paper['file']}) | [ACL Anthology]({paper['url']}) |")
    readings += ['\n## דוגמאות, הנחיות והצעה\n']
    for filename in PDFS:
        readings.append(f'- [{filename}](<../{filename}>)')
    readings += ['\n## קישורי עזר\n',
                 '- [מדריך כתיבת מאמרים של Vered Shwartz](https://medium.com/@vered1986/tips-for-writing-nlp-papers-9c729a2f9e1f) — מקור מקוון נוסף; לא הועתק מאחורי מגבלת גישה.',
                 '- [מקור mLAMA](https://github.com/norakassner/mlama) — מקור הנתונים. המידע השמור על האיסוף והייחוס נמצא בחבילה.',
                 '\nעותקי PDF נשמרו ללא שינוי. רישוי וזכויות היוצרים נשארים אצל בעלי המקורות; הקבצים מצורפים לעבודת צוות הקורס.\n']
    write('references/READING_LIST.md', '\n'.join(readings))

    blocks = re.findall(r'```markdown\n(.*?)```', guides[label], re.S)
    if len(blocks) != 2:
        raise ValueError('Guide note templates changed; inspect before packaging')
    prefix = language.upper()
    write(f'handoff/{prefix}_REVIEW_NOTES.md', blocks[0])
    write(f'handoff/{prefix}_ANALYSIS_NOTES.md', '# חומר ניתוח — למילוי אחרי קבלת תוצאות אמיתיות\n\n' + blocks[1])
    write(f'handoff/{prefix}_LITERATURE_NOTES.md',
          f'# הערות ספרות — {display}\n\nבודק/ת:\nתאריך:\n\n'
          '| מאמר וקישור | עמוד/סעיף/טבלה | טענה בפרפרזה | מודל ושפה | הראיה | מגבלה | שימוש בפרויקט | אומת במאמר מלא? |\n'
          '|---|---|---|---|---|---|---|---|\n\n'
          'יש למלא רק לאחר קריאה. אין כאן טענות מאושרות מראש.\n')
    write('handoff/RETURN_TO_ReviewerA.md',
          '# מה להחזיר למייס\n\n'
          f'1. קובץ הביקורת [{workbook_name}](../{workbook_relative}) לאחר שמירה.\n'
          f'2. [{prefix}_REVIEW_NOTES.md]({prefix}_REVIEW_NOTES.md) עם מקורות, החלטות וחסמים.\n'
          f'3. [{prefix}_LITERATURE_NOTES.md]({prefix}_LITERATURE_NOTES.md) ככל שהקריאה הושלמה.\n'
          f'4. [{prefix}_ANALYSIS_NOTES.md]({prefix}_ANALYSIS_NOTES.md) רק לאחר קבלת תוצאות ממייס.\n\n'
          'מייס משלבת את שני עותקי הביקורת לפי מזהים ועמודות אחריות. אין להחליף את הקובץ של השותף בעותק שלך. '
          'קובצי ביקורת התשובות והפלטים המחקריים יימסרו בהמשך; הם עדיין אינם קיימים בחבילה.\n')

    guide = f'docs/partner_guides/{label.upper()}_PARTNER_GUIDE.md'
    start = f'''# התחילו כאן — חבילת עבודה: {display}

**תאריך הכנה: 20 בספטמבר 2026.** זו חבילת העבודה של האחראי/ת על {display}. כל ההרצות על Aya ובשרת TAU וכתיבת המאמר נשארות אצל מייס.

1. חלצו את כל קובץ ה־ZIP לתיקייה רגילה במחשב. אין לערוך את Excel מתוך חלון ה־ZIP.
2. פתחו את [המדריך האישי]({guide}). התחילו במשימות של סעיפים 5–11.
3. פתחו את [קובץ הביקורת האישי]({workbook_relative}). עותק זה כבר קיבל שם נפרד לשפה; אין צורך להתחיל מחדש מקובץ אחר.
4. שמרו את ההערות ב־[יומן הביקורת](handoff/{prefix}_REVIEW_NOTES.md).
5. התחילו בחמש דוגמאות, תיאמו כללי בדיקה, והמשיכו לעבר 50 ישויות משותפות מאושרות בשתי השפות.

**תחום האחריות:** כל שמות הנושא והתשובה ב{display}, זוגות הניקוד ונוסחי השאלות בשפה; בדיקת מקורות ראשית ל־{relations}, והנוסחים באנגלית של קשרים אלה. פירוט עמודות ודוגמאות נמצא במדריך. אין לשנות את שמות הגיליונות או למחוק גיליון שפה: הקישורים והייבוא משתמשים במבנה המלא.

## איפה נמצא כל דבר

| פריט | פתיחה |
|---|---|
| מדריך עבודה מפורט | [המדריך]({guide}) |
| גיליון להזנת הביקורות | [Excel]({workbook_relative}) |
| מקורות לכל 70 הדוגמאות | [רשימת מקורות](references/PILOT_SOURCES.md) |
| מאמרים מקוריים וחומרי הקורס | [רשימת קריאה](references/READING_LIST.md) |
| תוכנית מחקר מלאה | [תוכנית](project_plan/RESEARCH_PLAN.md) |
| מצב המחקר והתשתית | [מצב העבודה](docs/STATUS.md) |
| מה להחזיר למייס | [הוראות מסירה](handoff/RETURN_TO_ReviewerA.md) |

## גבולות החבילה

ה־Excel הועתק בדיוק מהמקור, עם כל חמשת הגיליונות וכל המזהים. לא ניתנו אישורים ולא הוזנו ניקודים במהלך האריזה. נתוני המקור והקבצים בתיקיית data הם חומר עזר לשמירת ראיות; את העבודה מזינים ב־Excel ובקובצי handoff בלבד.

החבילה מכילה 50 דוגמאות Primary ועוד 20 Reserve ו־125 עותקי Wikidata שמורים לישויות ולמקומות שלהן. את המקורות החיצוניים המצוטטים ב־Wikidata ואת עדכניותם עדיין צריך לבדוק באינטרנט. אין בה פלטים מהניסויים המחקריים המלאים, משום שהם טרם בוצעו. דוחות Aya המצורפים הם בדיקות תשתית בלבד.

תוכנית המחקר וההצעה כוללות תכנון היסטורי ויעדים שעדיין לא הושגו. המדריך האישי ומצב העבודה הם נקודת הייחוס לחלוקת האחריות הנוכחית. אין לשנות חלוקת pilot/dev/test או לאשר רשומות כדי להגיע למכסה.

## קובצי מקור וייחוס

כל שותף מקבל את אותם מזהי דוגמאות וקובץ ביקורת התחלתי. העותק השמור של קובץ המקור מתועד ב־[מניפסט החבילה](PACKAGE_MANIFEST.json). חומרי הקורס והמאמרים המקוריים נשמרו ללא שינוי. המקורות של נתוני mLAMA מתועדים ב־[רשומת האיסוף](data/raw/mlama1.1.zip.source.json); הארכיון המלא נשאר אצל מייס, והרשומות הדרושות מצורפות בחבילה.

קובץ [הצהרת ההצעה המקורית](<NLP_Project_Proposal (1).pdf>) כולל את פרטי חברי הצוות כפי שנמסרו בקובץ המקורי. החבילה מיועדת לשיתוף עם שותפי הפרויקט.
'''
    write('START_HERE.md', start)
    links = [('מדריך העבודה', guide), ('פתיחת קובץ Excel', workbook_relative),
             ('התחילו כאן — פירוט', 'START_HERE.md'), ('מקורות הדוגמאות', 'references/PILOT_SOURCES.md'),
             ('מאמרי הקריאה', 'references/READING_LIST.md'), ('תיקיית המסירה', 'handoff/RETURN_TO_ReviewerA.md')]
    html_links = ''.join(f'<li><a href="{html.escape(path, quote=True)}">{html.escape(title)}</a></li>' for title, path in links)
    write('START_HERE.html', '<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8">'
          f'<title>חבילת {display}</title><style>body{{font:20px system-ui;max-width:850px;margin:48px auto;padding:20px;line-height:1.7}}'
          'li{margin:16px 0}a{color:#075985}</style>'
          f'<h1>חבילת עבודה — {display}</h1><p>חלצו את כל החבילה לפני פתיחת הקבצים. התחילו במדריך ובקובץ הביקורת האישי.</p>'
          f'<ul>{html_links}</ul><p>אם הדפדפן אינו מציג קובץ Markdown, פתחו אותו בעורך טקסט. '
          'את קובץ הביקורת אפשר לפתוח ישירות ב־Excel מתוך תיקיית החבילה.</p>'
          '<p>מייס אחראית להרצות על Aya ובשרת ולכתיבת המאמר. העבודה שלכם מתבצעת בקובצי הביקורת והמסירה.</p></html>')

    for path in sorted(target.rglob('*')):
        if path.is_file():
            relative = path.relative_to(target).as_posix()
            provenance[relative].update(bytes=path.stat().st_size, sha256=digest(path))
    manifest = {'package': target.name, 'language': language,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'workbook_path': workbook_relative, 'source_workbook_sha256': SOURCES[workbook_source],
                'notes': 'Copies and package-specific text only; no human reviews or research results added.',
                'files': provenance}
    # The manifest describes payload files, not its own checksum.
    (target / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return target, manifest


def verify_folder(folder, manifest):
    issues = []
    for relative, record in manifest['files'].items():
        path = folder / relative
        if digest(path) != record['sha256']:
            issues.append('hash: ' + relative)
    for file in folder.rglob('*.md'):
        text = file.read_text(encoding='utf-8')
        # Inspect only Markdown links, not future output names in backticks.
        for target in re.findall(r'\]\((<[^>]+>|[^)]+)\)', text):
            target = target.strip('<>')
            if target.startswith(('https://', 'http://', '#', 'mailto:')):
                continue
            path = (file.parent / target.split('#')[0]).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.exists():
                issues.append(f'link: {file.relative_to(folder)} -> {target}')
    for file in folder.rglob('*.pdf'):
        blob = file.read_bytes()
        if not blob.startswith(b'%PDF-') or b'%%EOF' not in blob[-2048:]:
            issues.append('pdf: ' + str(file.relative_to(folder)))
    workbook = folder / manifest['workbook_path']
    if digest(workbook) != manifest['source_workbook_sha256']:
        issues.append('workbook source mismatch')
    with zipfile.ZipFile(workbook) as zf:
        if zf.testzip() is not None:
            issues.append('invalid workbook archive')
        required = [f'xl/worksheets/sheet{i}.xml' for i in range(1, 6)]
        if any(name not in zf.namelist() for name in required):
            issues.append('missing workbook sheets')
    if issues:
        raise RuntimeError(json.dumps(issues, ensure_ascii=False, indent=2))


def archive(destination, folders):
    expected = {}
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for folder in folders:
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    name = folder.name + '/' + path.relative_to(folder).as_posix()
                    zf.write(path, name)
                    expected[name] = digest(path)
    with zipfile.ZipFile(destination) as zf:
        if zf.testzip() is not None or set(zf.namelist()) != set(expected):
            raise RuntimeError('ZIP content/integrity error: ' + str(destination))
        for name, checksum in expected.items():
            if hashlib.sha256(zf.read(name)).hexdigest() != checksum:
                raise RuntimeError('ZIP byte mismatch: ' + name)
    return {'file': destination.name, 'files': len(expected), 'bytes': destination.stat().st_size,
            'sha256': digest(destination), 'all_entries_verified': True}


if __name__ == '__main__':
    output = BASE
    if output.exists():
        output = BASE.with_name(BASE.name + '-' + datetime.now().strftime('%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    packet = json.loads(source_bytes('data/review/pilot_batch_v1/workbook_input.json'))
    papers = json.loads(source_bytes('data/reference_papers/download_manifest.json'))
    folders, archives = [], []
    for args in LANGUAGES:
        folder, manifest = make_package(output, *args, packet, papers)
        verify_folder(folder, manifest)
        folders.append(folder)
        print(json.dumps({'folder_verified': folder.name, 'files': len(manifest['files']) + 1}, ensure_ascii=False), flush=True)
        archives.append(archive(output / (folder.name + '.zip'), [folder]))
    archives.append(archive(output / 'Both_Partners.zip', folders))
    if any(digest(ROOT / relative) != checksum for relative, checksum in SOURCES.items()):
        raise RuntimeError('An original project source changed during packaging')
    report = {'status': 'verified', 'output_directory': str(output),
              'original_sources_unchanged': len(SOURCES), 'archives': archives}
    (output / 'PACKAGING_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
