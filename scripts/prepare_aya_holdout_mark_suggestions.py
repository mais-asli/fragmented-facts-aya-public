"""Write provisional human-checkable vocalizations; never mark them approved."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.unicode import validate_pair  # noqa: E402

BATCH = ROOT / "data" / "review" / "heldout_batch_v1"

# Fixed before any Aya test output. Deliberately a shortlist, not an inclusion
# decision: every one of the 60 test candidates still requires review.
# These are pronunciation proposals, not linguistic facts or human approvals.
SUGGESTIONS = {
    1: ("וִילְיָאם קַרְלוֹס וִילְיָאמְס", "وِيلْيَامْ كَارْلُوسْ وِيلْيَامْزْ"),
    2: ("לוּץ הֶק", "لُوتْز هِيك"),
    3: ("קְווֶנְטִין בֶּל", "كْوِينْتِن بِيل"),
    5: ("מַארְק סְטְרוֹנְג", "مَارْك سْتْرُونْج"),
    6: ("חַסְדַּאי קְרֶשְׂקַשׂ", "هَاسْدَاي كْرِيسْكَاس"),
    7: ("פְּיֶיר בְּרַאוּנְבֶּרְגֶר", "بِييَر بْرُونْبَرْجَر"),
    8: ("פְרַנְסֶס רַיְיט", "فْرَانْسِيس رَايْت"),
    9: ("אַנְגֶלָה בֵּיילִי", "أَنْجِيلَا بَيْلِي"),
    10: ("מֶהְמֶט הַשִּׁישִׁי", "مُحَمَّد السَّادِس الْعُثْمَانِي"),
    11: ("נִיקִי בַּארְנְס", "نِيكِي بَارْنْز"),
    12: ("סַ'עְבּ סַאלַאם", "صَائِب سَلَام"),
    13: ("רוֹי אַבֶּרְנָתִי", "رُوي أَبِيرْنِيثِي"),
    14: ("אַלֶכְּסַנְדֶּר הֶרְצֶן", "أَلِكْسَنْدَر هِيرْزِن"),
    15: ("קַרוֹלַיְין בַּיְינוּם", "كَارُولِين وُوكَر بَايْنُوم"),
    16: ("אַלְפוֹנְסוֹ הָרְבִיעִי, מֶלֶךְ אֲרָאגוֹן", "أَلْفُونْسُو الرَّابِع مَلِك أَرَاغُون"),
    17: ("אִבְּרָאהִים פָּאשָׁא", "إِبْرَاهِيم مُحَمَّد عَلِي بَاشَا"),
    19: ("וִילְיָאם בַּרְטוֹן רוֹגְ'רְס", "وِيلْيَام بَارْتُون رُوجَرْس"),
    20: ("יוֹהַאן פַּכֶלְבֶּל", "يُوهَان بَاتْشِيلْبِيل"),
    21: ("לוּאִי מַרְשָׁאן", "لُوِيس مَارْشَان"),
    22: ("אַנָה גּוֹנְזָאגָה", "آنْ غُونْزَاغَا"),
    23: ("גּוֹפְרֶדוֹ פֶּטְרָאסִי", "غُوفْرِيدُو بَاتْرَاسِي"),
    24: ("יָאן סְווַמֶרְדָם", "يَان زْفَامِرْدَام"),
    27: ("אַנָה מוֹרַנְדִּי מַנְזוֹלִינִי", "آنَّا مُورَانْدِي مَانْزُولِينِي"),
    28: ("וִילְיָאם חַאוִוי", "وِلِيم حَاوِي"),
    29: ("סַמוּאֵל אָדַמְס", "صَامُوِيل آدَمْز"),
    30: ("טוֹפִיק בַּחְרָאמוֹב", "تَوْفِيق بَهْرَامُوف"),
    32: ("אוּנִיבֶרְסִיטַת נוֹרְתְ'אִיסְטֶרְן", "جَامِعَة نُورْث إِيسْتِرْن"),
    33: ("דּוּפּוֹנְט", "دُو بُونْت"),
    35: ("אוּנִיבֶרְסִיטַת דֶּטְרוֹיְט מֶרְסִי", "جَامِعَة دِيتْرُويْت مِرْسِي"),
    39: ("אוּנִיבֶרְסִיטַת וַאנְדֶרְבִּילְט", "جَامِعَة فَانْدَرْبِيلْت"),
    41: ("הַמָּכוֹן לְלִימּוּדִים בֵּינְלְאוּמִיִּים מִתְקַדְּמִים", "الْمَعْهَد الْعَالِي لِلدِّرَاسَات الدَّوْلِيَّة وَالتَّنْمِيَة"),
    42: ("מִשְׁמְרוֹת הַמַּהְפֵּכָה הָאִסְלָאמִית", "الْحَرَس الثَّوْرِيّ الْإِسْلَامِيّ"),
    43: ("בִּימָאן בַּנְגְלָדֶשׁ אַיְירְלַיְינְס", "خُطُوط بِيمَان بَنْغْلَادِيش الْجَوِّيَّة"),
    45: ("הַשְּׂמֹאל הָרֶפּוּבְּלִיקָנִי שֶׁל קַטָלוֹנְיָה", "الْيَسَار الْجُمْهُورِيّ لِكَتَالُونْيَا"),
    46: ("מוֹסְפִילְם", "مُوسْفِيلْم"),
    47: ("חֶבְרַת קוֹקָה-קוֹלָה", "شَرِكَة كُوكَا كُولَا"),
    49: ("פַּרַאגָאמוֹ", "فِيرَاغَامُو"),
    50: ("אֶלֶקְטְרוֹלוּקְס", "إِلِكْتْرُولُوكْس"),
    51: ("סַן מִיגֶל", "سَان مِيغِيل كُورْبُورَيْشِن"),
    52: ("אַרְמָנִי (חֶבְרָה)", "أَرْمَانِي"),
    56: ("פֵייֶנוֹרְד", "فَايْنُورْد"),
    57: ("פִּיקְסִיז", "بِيكْسِيز"),
    58: ("סְווֶנְדַּאסְט", "سِيفِينْدَاسْت"),
    59: ("אַרְקְטִיק מַאנְקִיז", "أَرْكْتِك مُونْكِيز"),
}

out = BATCH / "AI_MARK_SUGGESTIONS_NOT_APPROVED_20260927.csv"
with out.open("w", encoding="utf-8-sig", newline="") as fh:
    columns = ["review_order", "fact_id", "subject_he_U_candidate", "subject_he_D_ai_suggestion",
               "subject_ar_U_candidate", "subject_ar_D_ai_suggestion", "machine_pair_valid",
               "human_review_status", "ai_note"]
    writer = csv.DictWriter(fh, fieldnames=columns)
    writer.writeheader()
    for row in csv.DictReader((BATCH / "AYA_ONLY_TEST_REVIEW_QUEUE_20260927.csv").open(encoding="utf-8-sig", newline="")):
        order = int(row["review_order"])
        he_d, ar_d = SUGGESTIONS.get(order, ("", ""))
        valid = "not_suggested"
        note = "Check pronunciation, identity and source; all proposals require ReviewerA's review."
        if he_d and ar_d:
            try:
                validate_pair(row["subject_he_U_candidate"], he_d, "he")
                validate_pair(row["subject_ar_U_candidate"], ar_d, "ar")
                valid = "yes"
            except ValueError as exc:
                valid = "no"
                note = f"Revise base letters/marks before use: {exc}. " + note
        writer.writerow({"review_order": order, "fact_id": row["fact_id"],
                         "subject_he_U_candidate": row["subject_he_U_candidate"],
                         "subject_he_D_ai_suggestion": he_d,
                         "subject_ar_U_candidate": row["subject_ar_U_candidate"],
                         "subject_ar_D_ai_suggestion": ar_d,
                         "machine_pair_valid": valid, "human_review_status": "unreviewed",
                         "ai_note": note})
print(out)
