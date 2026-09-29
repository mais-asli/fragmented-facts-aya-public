"""Prepare unapproved Hebrew niqqud candidates for the 70-row pilot review.

The strings are proposals. Unicode identity is machine-checked; pronunciation
and entity naming still require a human language decision.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.unicode import validate_pair

SOURCE = ROOT / "data/review/merged_20260925/merged_input.json"
DEST = ROOT / "data/review/merged_20260925/hebrew_mark_proposals.json"

# In the frozen workbook order. Keeping the English identity beside each entry
# prevents accidental reassignment if the underlying review order ever changes.
PROPOSALS = [
    ("Ramil Guliyev", "רָמִיל גוּלִיֶיב"),
    ("Peter Parler", "פֶּטֶר פָּארְלֶר"),
    ("Hang Seng Bank", "בַּנְק הַאנְג סֶנְג"),
    ("Singer Motors", "מְנוֹעֵי סִינְגֶר"),
    ("Willebrord Snellius", "וִילְבְּרוֹרְד סְנֶל"),
    ("Michael Wolgemut", "מִיכָאֵל ווֹלְגֶמוּט"),
    ("European Space Agency", "סוֹכְנוּת הֶחָלָל הָאֵירוֹפִּית"),
    ("Nation of Gods and Earths", "אוּמַת חֲמֵשֶׁת הָאֲחוּזִים"),
    ("Qasim Amin", "קָאסֶם אָמִין"),
    ("Faiz Ahmad Faiz", "פַאִיז אַחְמַד פַאִיז"),
    ("Chivas Regal", "שִׁיבַס רִיגַל"),
    ("Nikon", "נִיקוֹן"),
    ("Yvonne Blake", "אִיבוֹנָה בְּלֵייק"),
    ("Karl Liebknecht", "קַרְל לִיבְּקְנֶכְט"),
    ("Croatian Democratic Union", "הָאִיחוּד הַדֵּמוֹקְרָטִי הַקְּרוֹאַטִי"),
    ("The White Stripes", "הַווֵייט סְטְרַייפְּס"),
    ("Alexander Cochrane", "אֲלֶכְּסַנְדֶּר קוֹקְרֶן"),
    ("Gertrude Bell", "גֶּרְטְרוּד בֶּל"),
    ("Aigle Azur", "אֶגְל אָזוּר"),
    ("Great Western Railway", "גְּרֵייט ווֶסְטֶרְן רֵיילְווֵיי"),
    ("John Addington Symonds", "ג'וֹן אָדִינְגְטוֹן סִימוֹנְדְס"),
    ("Carl Ritter", "קַרְל רִיטֶר"),
    ("McGill University", "אוּנִיבֶרְסִיטַת מֶקְגִיל"),
    ("Foo Fighters", "פוּ פַייטֶרְס"),
    ("Maria Altmann", "מַרְיָה אַלְטְמַן"),
    ("John Selden", "ג'וֹן סֶלְדֶן"),
    ("Toyota", "טוֹיוֹטָה"),
    ("Aerosvit Airlines", "אַאֵרוֹסְוִויט"),
    ("Perry Benson", "פֶּרִי בֶּנְסוֹן"),
    ("Michelangelo", "מִיכֶּלְאַנְג'לוֹ"),
    ("Dresdner SC", "מוֹעֲדוֹן סְפּוֹרְט דְּרֶזְדֶן"),
    ("Chunghwa Telecom", "טְשוֹנְגְווָאה טֶלֶקוֹם"),
    ("Diane Hendricks", "דִּיאַן הֶנְדְרִיקְס"),
    ("Gregory XVI", "גְּרֶגוֹרְיוּס הַשִּׁישָּׁה עָשָׂר"),
    ("Partick Thistle F.C.", "פַּרְטִיק תִ'יסְל"),
    ("Ducati Motor Holding S.p.A.", "דוּקָאטִי"),
    ("William Thomson, 1st Baron Kelvin", "וִילְיָאם תּוֹמְסוֹן"),
    ("Girolamo Frescobaldi", "גִ'ירוֹלָאמוֹ פְרֶסְקוֹבַּלְדִּי"),
    ("Obsidian Entertainment", "אוֹבְסִידִיאַן אֶנְטֶרְטֵיינְמֶנְט"),
    ("Girls' Generation", "גִּירְלְז גֶ'נֵרֵיישֶׁן"),
    ("Tobias Rau", "טוֹבִּיאָס רָאוּ"),
    ("Princess Milica of Montenegro", "מִילִיצָה, נְסִיכַת מוֹנְטֶנֶגְרוֹ"),
    ("Chinese Academy of Sciences", "הָאָקָדֶמְיָה הַסִּינִית לְמַדָּעִים"),
    ("Eni", "אֶנִי (חֶבְרָה)"),
    ("Dennis Farina", "דֶּנִיס פָרִינָה"),
    ("Giovanni Pico della Mirandola", "ג'וֹבָאנִי פִּיקוֹ דֶלָה מִירַנְדוֹלָה"),
    ("Indian Space Research Organisation", "הָאִרְגּוּן הַהוֹדִי לְחֵקֶר הֶחָלָל"),
    ("Opeth", "אוֹפֶּת'"),
    ("Arthur Harden", "אַרְתוּר הַרְדֶן"),
    ("Brian O'Nolan", "בְּרַיְאָן אוֹנוֹלָן"),
    ("Deutsche Reichsbahn (GDR)", "דּוֹיטְשֶׁה רַייכְסְבָּאן (גֶּרְמַנְיָה הַמִּזְרָחִית)"),
    ("Siemens", "סִימֶנְס"),
    ("Kurupt", "קוּרוּפְּט"),
    ("Jack Cassidy", "גֵ'ק קָסִידִי"),
    ("University of South Florida", "אוּנִיבֶרְסִיטַת דְּרוֹם פְלוֹרִידָה"),
    ("Samsung Electronics", "סָמְסוּנְג אֶלֶקְטְרוֹנִיקָה"),
    ("Francesco Foscari", "פְרַנְצֶ'סְקוֹ פוֹסְקָארִי"),
    ("Giorgione", "ג'וֹרְג'וֹנֶה"),
    ("Hapoel Haifa F.C.", "הַפּוֹעֵל חֵיפָה"),
    ("Institut d'Estudis Catalans", "הַמּוֹסָד לְלִימּוּדִים קָטָלָאנִיִּים"),
    ("Placido Costanzi", "פְּלָסִידוֹ קוֹנְסְטַנְצִי"),
    ("Juan Prim, 1st Marquis of los Castillejos", "חוּאָן פְּרִים"),
    ("Swindon Town F.C.", "סְווִינְדוֹן טָאוּן"),
    ("Greenpeace", "גְּרִינְפִּיס"),
    ("Brendan Behan", "בְּרֶנְדֶן בֶּהָאן"),
    ("Yehoshafat Harkabi", "יְהוֹשָׁפָט הַרְכָּבִּי"),
    ("FIBA Africa", "פִיבָּ\"א אַפְרִיקָה"),
    ("scouting", "תְּנוּעַת הַצּוֹפִים"),
    ("Bruno Frank", "בְּרוּנוֹ פְרַנְק"),
    ("Joseph Henry Blackburne", "ג'וֹזֶף הֶנְרִי בְּלֶקְבּוֹרְן"),
]

U_CORRECTIONS = {
    "Obsidian Entertainment": {
        "U": "אובסידיאן אנטרטיינמנט",
        "reason": "Existing Hebrew label is English; the company name is proposed in Hebrew, extending the attested אובסידיאן shorthand. Full pronunciation needs human confirmation.",
        "source": "https://gamerspack.co.il/2017/01/23/אובסידיאן-משחררת-טיזר-לפרוייקט-לואיזיאנה/",
    }
}


def main():
    merged = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = merged["Hebrew"]
    if len(rows) != len(PROPOSALS):
        raise ValueError("Review row count changed")
    result = []
    for row, (name, marked) in zip(rows, PROPOSALS):
        if row["English name"] != name:
            raise ValueError(f"Unexpected review order: {row['English name']} != {name}")
        correction = U_CORRECTIONS.get(name)
        unmarked = correction["U"] if correction else row["Unmarked spelling"]
        try:
            validate_pair(unmarked, marked, "he")
        except ValueError as exc:
            raise ValueError(f"{name}: U={unmarked!r}, D={marked!r}: {exc}") from exc
        result.append({"fact_id": row["Fact ID"], "English name": name,
                       "U": unmarked, "D": marked, "source": correction["source"] if correction else None,
                       "note": correction["reason"] if correction else "Pronunciation proposal; native review pending.",
                       "decision": "pending"})
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(result), "unicode_valid": len(result),
                      "canonical_hebrew_label_corrections": list(U_CORRECTIONS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
