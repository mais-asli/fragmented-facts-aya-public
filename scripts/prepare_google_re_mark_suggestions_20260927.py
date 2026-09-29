"""Add unapproved Hebrew/Arabic mark suggestions to the external review copy.

These are editorial proposals. The frozen source-selection CSV stays unchanged.
"""

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.unicode import validate_pair  # noqa: E402

BASE = ROOT / 'data/review/google_re_external_20260927'
SOURCE = BASE / 'CANDIDATES_32_GOOGLE_RE_20260927.csv'
OUT = BASE / 'CANDIDATES_32_WITH_MARK_SUGGESTIONS_20260927.csv'
REPORT = BASE / 'MARK_SUGGESTIONS_MANIFEST_20260927.json'

MARKS = {
    'Matteo Salvini': ('מַתֵּאוֹ סַלְבִינִי', 'مَاتْيُو سَالْفِينِي'),
    'Tom Arnold': ('טוֹם אַרְנוֹלְד', 'طُوم أَرْنُولْد'),
    'Mary Jayne Gold': ('מֵרִי גֵ\'יְין גוֹלְד', 'مَارِي جُولْد'),
    'René Mayer': ('רֶנֶה מַאיֶיר', 'رِينِيه مَايَر'),
    'John Barry': ('ג\'וֹן בַּרִי', 'جُون بَارِي'),
    'Lev Leshchenko': ('לֵב לֶשְׁצֶ\'נְקוֹ', 'لِيف لِيشِينْكُو'),
    'Marcel Bertrand': ('מַרְסֶל אַלֶכְּסַנְדֶר בֶּרְטְרַאן', 'مَارْسِيل أَلِكْسَنْدَر بَرْتْرَانْد'),
    'Eduard Ender': ('אֶדוּאַרְד אֶנְדֶר', 'إِدْوَارْد إِنْدَر'),
    'James Burnham': ('גֵ\'יְימְס בֶּרְנְהַם', 'جِيمْس بِيرْنْهَام'),
    'Bernard Brodie': ('בֶּרְנַארְד בְּרוֹדִי', 'بِرْنَارْد بْرُودِي'),
    'Don Mankiewicz': ('דּוֹן מַנְקְיֵיבִיץ', 'دُون مَانْكِيفِيتْس'),
    'Staffan de Mistura': ('סְטַפָן דֶה מִיסְטוּרָה', 'سْتَافَان دِي مِيسْتُورَا'),
    'Dan Armon': ('דָּן עַרְמוֹן', 'دَان أَرْمُون'),
    "John O'Conor": ("ג'וֹן אוֹ'קוֹנוֹר", 'جُون أُوكُونُور'),
    'Leo Alexander': ('לֵאוֹ אַלֶכְּסַנְדֶר', 'لِيو أَلِكْسَنْدَر'),
    'Richard Anthony': ('רִישַׁאר אַנְטוֹנִי', 'رِيتْشَارْد أَنْتُونِي'),
    'Lotfia Elnadi': ('לוּטְפִיָה אַ-נַאדִי', 'لُطْفِيَّة النَّادِي'),
    'Elisha Netanyahu': ('אֱלִישָׁע נְתַנְיָהוּ', 'إِلِيشَا نَتَنْيَاهُو'),
    'Eliezer Waldenberg': ('אֱלִיעֶזֶר יְהוּדָה וַלְדֶנְבֶּרְג', 'إِلِيعِيزَر وَالْدِنْبِرْغ'),
    'Sebastiano Baggio': ('סֶבַּסְטְיָאנוֹ בַּאג\'וֹ', 'سِيبَاسْتِيَانُو بَاجِيُو'),
    'Wolf Gold': ('זְאֵב גוֹלְד', 'وُولْف جُولْد'),
    'Robert Hazard': ('רוֹבֶּרְט הַזַאר', 'رُوبِرْت هَازَارْد'),
    'Simon Halkin': ('שִׁמְעוֹן הַלְקִין', 'شِمْعُون هَالْكِين'),
    'Christa Wolf': ('כְּרִיסְטָה ווֹלְף', 'كْرِيسْتَا فُولْف'),
    'Moshe Lewin': ('מֹשֶׁה לֵוִין', 'مُوشِيه لِيفِين'),
    'David Merrick': ('דֵּיוִיד מַארִיק', 'دِيفِيد مِيرِيك'),
    'Aylmer Haldane': ('אַיְילְמֶר הַלְדֵיְין', 'أَيْلْمَر هَالْدِين'),
    'Johan Gustaf Sandberg': ('יוֹהַאן גוּסְטַב סַנְדְבֶּרְג', 'جُون غُوسْتَاف سَانْدْبِرْغ'),
    'Abdülmecid I': ('אַבְּדִילְמַג\'ִיט הָרִאשׁוֹן', 'عَبْدُ الْمَجِيدِ الْأَوَّل'),
    'Peter Norman Nissen': ('פִּיטֶר נוֹרְמַן נִיסֶן', 'بِيتَر نُورْمَان نِيسِن'),
    'Alexander Guchkov': ('אַלֶכְּסַנְדֶר גוּצ\'ְקוֹב', 'أَلِكْسَنْدَر غُوتْشْكُوف'),
    'Francis Alexander': ('פְרַנְסִיס אַלֶכְּסַנְדֶר', 'فْرَنْسِيس الْكْسَنْدَر'),
}

FLAG = {
    'Mary Jayne Gold': 'Arabic U omits Jayne; revise the base spelling and D together or reject this language pair.',
    'Eliezer Waldenberg': 'Arabic U omits Yehuda; confirm whether the shorter common name preserves identity.',
    'Johan Gustaf Sandberg': 'Arabic U says جون for Johan; verify identity/pronunciation before approval.',
    'Richard Anthony': 'Check French pronunciation against Arabic transliteration.',
    'Francis Alexander': 'Arabic الكسندر lacks the hamza in ألكسندر; preserve base letters unless U is corrected and re-reviewed.',
}


def main():
    if OUT.exists() or REPORT.exists():
        raise FileExistsError('External mark-suggestion review copy is already frozen')
    with SOURCE.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames
    if len(rows) != 32 or {r['subject_en_google_re'] for r in rows} != set(MARKS):
        raise ValueError('External mark suggestions no longer match frozen candidate identities')
    valid_count = 0
    for row in rows:
        name = row['subject_en_google_re']
        he, ar = MARKS[name]
        warnings = [FLAG[name]] if name in FLAG else []
        for language, suggestion in (('he', he), ('ar', ar)):
            try:
                validate_pair(row[f'subject_{language}_U_candidate'], suggestion, language)
            except ValueError as exc:
                suggestion = ''
                warnings.append(f'{language} suggestion withheld because base letters differ: {exc}')
            else:
                valid_count += 1
            row[f'subject_{language}_D_ai_suggestion'] = suggestion
        row['ai_mark_warning'] = ' '.join(warnings)
    with OUT.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns + [
            'subject_he_D_ai_suggestion', 'subject_ar_D_ai_suggestion', 'ai_mark_warning'])
        writer.writeheader()
        writer.writerows(rows)
    REPORT.write_text(json.dumps({
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'review_copy_sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
        'status': 'AI-suggested marks only; zero human approvals entered',
        'mechanically_valid_pairs': valid_count,
        'withheld_invalid_base_pairs': len(rows) * 2 - valid_count,
        'identity_or_pronunciation_warnings': FLAG,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(rows)} external facts with {valid_count} mechanically valid, unapproved marked-name suggestions')


if __name__ == '__main__':
    main()
