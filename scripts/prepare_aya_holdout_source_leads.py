"""Write research leads for the outcome-blind Aya test review (never approvals)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH = ROOT / "data" / "review" / "heldout_batch_v1"

# These are leads to check, not final source adjudications or human review.
LEADS = {
    "Q178106-P19-Q589182": ("https://njhalloffame.org/hall-of-famers/2009-inductees/william-carlos-williams/", "NJ Hall of Fame lists birth in Rutherford; verify city QID."),
    "Q73498-P19-Q64": ("https://kalliope-verbund.info/gnd/eac?eac.id=116560495", "Archival authority lists Lutz Heck birth in Berlin."),
    "Q649822-P19-Q84": ("https://www.independent.co.uk/news/people/obituary-professor-quentin-bell-1315047.html", "1996 obituary lists birth in London."),
    "Q312712-P19-Q84": ("https://www.markstrong.co.uk/", "Official actor site confirms identity; find a direct biographical source explicitly stating London birthplace."),
    "Q542741-P19-Q1781": ("https://www.micheledidier.com/cspdocs/exhibition/files/mfc_2014_yona_friedman_dp.pdf", "2014 exhibition biography lists Budapest; Arabic cached subject appears to say Yunus, so correct identity before use."),
    "Q528415-P19-Q1492": ("https://plato.stanford.edu/entries/crescas/", "Stanford Encyclopedia says Hasdai Crescas born Barcelona."),
    "Q343898-P19-Q90": ("https://www.lemonde.fr/archives/article/1987/12/25/un-livre-de-souvenirs-de-pierre-braunberger-memoires-d-un-producteur_4083200_1819218.html", "1987 article states born Paris."),
    "Q441537-P19-Q123709": ("https://scholarworks.iu.edu/journals/index.php/imh/article/download/7133/7980/0", "Historical journal article states Frances Wright born Dundee."),
    "Q272721-P19-Q6225": ("https://olympic.ca/team-canada/angela-bailey/", "Olympic profile lists Coventry; verify cached place QID."),
    "Q162543-P19-Q406": ("https://ttk.gov.tr/osmanli-padisahlari/", "Turkish Historical Society section on Mehmed Vahdeddin states born Istanbul on 2 February 1861; verify Hebrew and Arabic names specify Mehmed VI."),
    "Q693462-P19-Q189074": ("https://www.imdb.com/name/nm2752759/", "IMDb lists Harlem; verify whether Harlem district granularity matches gold."),
    "Q2002354-P19-Q3820": ("https://thedocs.worldbank.org/en/doc/956281391204516645-0560011973/original/WorldBankGroupArchivesFolder1772669.pdf", "1973 World Bank archive lists Saeb Salam born Beirut."),
    "Q201221-P19-Q649": ("https://www.gutenberg.org/files/67882/67882-h/67882-h.htm", "Published memoir introduction says Alexander Herzen born in Moscow."),
    "Q528840-P19-Q23556": ("https://professorsemeritus.columbia.edu/people/caroline-walker-bynum", "Columbia biography explicitly says Caroline Walker Bynum born in Atlanta; cached English subject omits Walker, but identity may match."),
    "Q380088-P20-Q1492": ("https://www.treccani.it/enciclopedia/alfonso-iv-d-aragona-iii-di-catalogna-detto-il-benigno_%28Enciclopedia-Italiana%29/", "Treccani lists death in Barcelona."),
    "Q311789-P20-Q85": ("https://en.wikisource.org/wiki/The_New_Student%27s_Reference_Work/Ibrahim_Pasha", "Historical reference lists Cairo death; verify exact Ibrahim Pasha identity."),
    "Q41689-P20-Q16869": ("https://www.biolex.ios-regensburg.de/BioLexViewview.php?ID=1452&export=print", "Biographical research resource says Nikephoros III died in Constantinople as a monk; verify full identity and source date."),
    "Q2341286-P20-Q100": ("https://cdn.libraries.mit.edu/dissemination/diponline/T171.M427/T171.M4217_Volume%202.pdf", "MIT-hosted 1896 biography records death at 1882 MIT commencement; confirm ceremony city Boston in the source."),
    "Q76485-P20-Q2090": ("https://aaa.gf-franken.de/de/recherche.html?permaLink=10881", "Historical scholarly record lists Johann Pachelbel death in Nuremberg."),
    "Q448822-P20-Q90": ("https://catalogue.bnf.fr/ark%3A/12148/cb138970711", "BnF authority lists Louis Marchand death in Paris."),
    "Q2556248-P20-Q90": ("https://www.treccani.it/enciclopedia/anna-gonzaga_%28Enciclopedia-Italiana%29/", "Treccani lists Anne Gonzaga death in Paris."),
    "Q505806-P20-Q220": ("https://siusa-archivi.cultura.gov.it/cgi-bin/siusa/pagina.pl?Chiave=54644&RicLin=en&RicSez=prodpersone&RicTipoScheda=pp&RicVM=indice&TipoPag=prodpersona", "Italian archive lists Goffredo Petrassi death in Rome."),
    "Q336794-P20-Q727": ("https://www.deutsche-biographie.de/pnd119475308.html?language=en", "Deutsche Biographie explicitly labels Jan Swammerdam's place of death Amsterdam."),
    "Q269701-P20-Q1891": ("https://www.treccani.it/enciclopedia/morandi-manzolini-anna/", "Treccani lists Anna Morandi Manzolini 1716–1774 Bologna."),
    "Q78906-P20-Q64": ("https://congressforjewishculture.org/people/267/Steinschneider-Moritz-March-30-1816-January-24-1907", "Congress for Jewish Culture states Moritz Steinschneider died in Berlin; Hebrew given name is Moshe, verify cached subject identity."),
    "Q4116097-P20-Q3820": ("https://elpais.com/diario/1976/07/14/internacional/206143210_850215.html", "1976 El Pais report locates William Hawi's fatal incident at Tel al-Zaatar in Beirut; review city vs district granularity."),
    "Q212963-P20-Q100": ("https://www.nps.gov/people/samuel-adams.htm", "US National Park Service biography explicitly states Samuel Adams died in Boston."),
    "Q448820-P20-Q9248": ("https://sportnews.az/en/football/33rd-anniversary-of-tofiq-bahramovs-death-is-commemorated", "Azerbaijani sports article says Tofiq Bahramov died in Baku in 1993; verify date and person identity."),
    "Q37548-P159-Q100": ("https://nupd.northeastern.edu/wp-content/uploads/NU-EOP-Boston-Annex.pdf", "2019 Northeastern document locates Boston campus; decide whether campus supports P159 headquarters."),
    "Q19568-P159-Q170478": ("https://find-and-update.company-information.service.gov.uk/company/06632170", "UK Companies House lists AFC Bournemouth Limited at Dean Court in Bournemouth; cached Hebrew is town-only and must be replaced with club name."),
    "Q221062-P159-Q174224": ("https://www.dupont.com/locations.html", "DuPont lists global headquarters in Wilmington; verify pre-May-2024 status."),
    "Q658637-P159-Q37836": ("https://staging.politifact.com/factchecks/2014/sep/08/paul-ryan/paul-ryan-says-miller-brewing-and-anheuser-busch-a/", "LIKELY REJECT: 2014 source discusses headquarters move from Milwaukee to Chicago; entity/time ambiguous."),
    "Q1320232-P159-Q12439": ("https://www.udmercy.edu/about/location.php", "University says main administration is at Detroit McNichols campus; verify headquarters interpretation."),
    "Q27436-P159-Q12439": ("https://www.cityofwarren.org/wp-content/uploads/2019/03/2019_Newsbeat_Spring.pdf", "LIKELY REJECT: Cadillac brand headquarters moved to Warren in 2019; Detroit is parent GM city."),
    "Q714335-P159-Q1867": ("https://www.benq.com/zh-tw/about-benq/globaloffice.html", "BenQ lists Taipei headquarters; Hebrew cached subject is Latin-only and needs a valid script-form pair."),
    "Q152057-P159-Q84": ("https://www.sec.gov/Archives/edgar/data/313807/000031380724000008/bp-20231231.htm", "BP 2023 filing gives London address; Hebrew cached subject is Latin-only and needs a valid pair."),
    "Q29052-P159-Q23197": ("https://www.vanderbilt.edu/campuses/", "Vanderbilt identifies Nashville main campus; verify P159 interpretation."),
    "Q3068744-P159-Q3766": ("https://www.lemonde.fr/en/international/article/2025/07/17/by-bombing-damascus-israel-imposes-red-lines-on-syria-s-new-government_6743462_4.html", "TIME/ENTITY CAUTION: Syrian military changed after December 2024; distinguish pre-2024 Syrian Arab Army from current entity and verify P159 headquarters."),
    "Q691686-P159-Q71": ("https://www.graduateinstitute.ch/venues/contact", "Geneva Graduate Institute contact page locates institute in Geneva; verify exact headquarters meaning."),
    "Q271110-P159-Q3616": ("https://apnews.com/article/239568283d236bc12e57a366fdc805d5", "CAUTION: 2023 AP report places IRGC aerospace-branch headquarters in western Tehran; branch address does not itself prove parent IRGC headquarters."),
    "Q863432-P159-Q1354": ("https://biman.gov.bd/site/page/e638096d-3eca-4de0-afd6-3b4a5220a350/-", "Bangladesh government Biman page lists head office in Dhaka."),
    "Q696056-P159-Q1055": ("https://www.hapag-lloyd.com/en/company/about-us/history.html", "LIKELY REJECT for current P159: original HAPAG merged into Hapag-Lloyd in 1970; do not substitute successor's Hamburg headquarters."),
    "Q150068-P159-Q1492": ("https://www.esquerra.cat/avis-legal/", "Republican Left of Catalonia legal notice names Barcelona party address at Carrer Calabria 166."),
    "Q141336-P740-Q649": ("https://www.mosfilm.ru/about/news/k-100-letiyu-kinostudii-mosfilm-istoriya-lyudi-tayny2/", "2023 Mosfilm history supports 1924 origin; seek explicit founding-city Moscow evidence."),
    "Q3295867-P740-Q23556": ("https://investors.coca-colacompany.com/news-events/press-releases/detail/831/coca-cola-marks-130th-anniversary-with-gift-to-atlantas-centennial-olympic-park-district", "2016 company release connects founding to Atlanta; distinguish company from 1886 drink."),
    "Q13646-P740-Q90": ("https://www.groupe-sncf.com/fr/groupe/patrimoine-archives/80-ans-histoire", "SNCF history says its original headquarters in Paris from creation in 1938; verify whether this supports founding place Paris. Cached Hebrew subject is Latin-only and cannot be vocalized."),
    "Q3946053-P740-Q2044": ("https://group.ferragamo.com/en/the-group/group-history", "Ferragamo history explicitly says first company based in Florence in 1927."),
    "Q505922-P740-Q1754": ("https://www.electroluxgroup.com/en/?p=26396", "Electrolux company timeline records 1919 formation; verify founding city Stockholm separately."),
    "Q934489-P740-Q1461": ("https://www.sanmiguel.com.ph/page/the-shape-of-the-future", "San Miguel history places first 1890 brewery in Manila; check company identity."),
    "Q635484-P740-Q490": ("https://www.armani.com/en-us/giorgio-armani/experience/about-giorgio-armani/", "Armani says brand founded Milan in 1975; check whether subject QID denotes same entity."),
    "Q625975-P740-Q8684": ("https://links.sgx.com/FileOpen/Project%20Elybely%20-%20Final%20OC%20%28dd%2022.04.24%29.ashx?App=Prospectus&FileID=62630", "2024 filing says Seoul Records predecessor of Kakao M was founded in 1978; does not directly establish founding city Seoul; cached Hebrew is Latin-only."),
    "Q269836-P740-Q8684": ("https://www.allmusic.com/artist/shinee-mn0002776527", "AllMusic lists SHINee formed in Seoul in 2008; cached Hebrew Shinee Latin-only, requires a reviewed Hebrew-script name."),
    "Q177814-P740-Q90": ("https://www.suez.com/-/media/suez-global/files/publication-docs/pdf-english/ra-2013-en.pdf", "2013 Suez Environnement report establishes company identity but Paris founding location needs direct evidence; cached Hebrew and Arabic are Latin-only."),
    "Q134241-P740-Q34370": ("https://www.feyenoord.com/nl/de-club/organisatie/historie", "Feyenoord history dates origin 1908 in Rotterdam; verify exact founding locality."),
    "Q188464-P740-Q100": ("https://www.allmusic.com/artist/pixies-mn0000895136/biography", "AllMusic explicitly says Pixies formed in Boston in January 1986."),
    "Q741801-P740-Q23556": ("https://label.napalmrecords.com/images/bands/sevendust/Bio%20-%20Sevendust%202026%20FINAL.pdf", "Label biography calls Sevendust Atlanta quintet since 1994; verify this is explicit founding place, not later base."),
    "Q170599-P740-Q42448": ("https://www.officialcharts.com/artist/17120/Arctic-Monkeys/", "Official Charts says Arctic Monkeys were founded in High Green, Sheffield; city-level Sheffield matches gold but note finer locality."),
    "Q231985-P740-Q79990": ("https://www.wikidata.org/wiki/Q231985", "REJECT: Hayley Westenra is a person, so organization founding-place question is inapplicable."),
}

out = BATCH / "AI_SOURCE_LEADS_NOT_APPROVED_20260927.csv"
with out.open("w", encoding="utf-8-sig", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["fact_id", "ai_source_lead_url", "ai_note", "human_review_status"])
    writer.writeheader()
    for row in csv.DictReader((BATCH / "AYA_ONLY_TEST_REVIEW_QUEUE_20260927.csv").open(encoding="utf-8-sig", newline="")):
        url, note = LEADS.get(row["fact_id"], ("", "No direct source lead yet; requires review."))
        writer.writerow({"fact_id": row["fact_id"], "ai_source_lead_url": url,
                         "ai_note": note, "human_review_status": "unreviewed"})
print(out)
