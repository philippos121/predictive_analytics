"""
generate_sample_data.py — Erzeugt synthetische Testdaten im erwarteten JSON-Format.

Nur für Tests gedacht! Die echten Daten kommen aus dem Data Extractor.

Verwendung:
  python generate_sample_data.py --count 100 --output sample_cases.json
"""

import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Vorlagen für synthetische Fälle (vereinfacht)
# ---------------------------------------------------------------------------

KLAEGER_TEMPLATES = [
    (
        "Der Kläger macht einen Schadenersatzanspruch in Höhe von EUR {betrag:,.2f} "
        "geltend. Er bringt vor, dass der Beklagte seine vertraglichen Pflichten "
        "aus dem {vertragstyp} vom {datum} schuldhaft verletzt hat, indem er "
        "{pflichtverletzung}. Der Kläger hat den Beklagten am {mahndatum} "
        "schriftlich zur Leistung aufgefordert."
    ),
    (
        "Der Kläger begehrt die Zahlung von EUR {betrag:,.2f} s.A. aus dem Titel "
        "des {rechtsgrund}. Der Beklagte hat am {datum} {sachverhalt}. "
        "Trotz Fristsetzung und Mahnung vom {mahndatum} ist der Beklagte seiner "
        "Zahlungsverpflichtung nicht nachgekommen."
    ),
    (
        "Der Kläger stützt sein Klagebegehren auf {rechtsgrund} und fordert "
        "EUR {betrag:,.2f}. Der Beklagte hat {pflichtverletzung}, wodurch dem "
        "Kläger ein Schaden in der geltend gemachten Höhe entstanden ist. "
        "Die Kausalität und Rechtswidrigkeit sind gegeben."
    ),
]

BEKLAGTER_TEMPLATES = [
    (
        "Der Beklagte bestreitet das Klagebegehren dem Grunde und der Höhe nach. "
        "Er wendet ein, dass {einwendung}. Zudem sei {verjährung_einrede}. "
        "Der Beklagte beantragt die Abweisung der Klage."
    ),
    (
        "Der Beklagte bringt vor, dass er seine vertraglichen Pflichten ordnungsgemäß "
        "erfüllt hat. {gegenargument}. Ein Verschulden des Beklagten liegt nicht vor, "
        "da {entlastung}. Das Klagebegehren ist daher abzuweisen."
    ),
    (
        "Der Beklagte bestreitet die klägerischen Behauptungen und wendet ein, "
        "dass {einwendung}. Selbst bei Vorliegen einer Pflichtverletzung sei "
        "dem Kläger ein Mitverschulden von mindestens {mitverschulden}% anzulasten, "
        "da {mitverschulden_grund}."
    ),
]

VERTRAGSTYPEN = [
    "Kaufvertrag", "Werkvertrag", "Mietvertrag", "Dienstvertrag",
    "Darlehensvertrag", "Gesellschaftsvertrag", "Lizenzvertrag",
    "Generalunternehmervertrag", "Bestandvertrag",
]

RECHTSGRUENDE = [
    "ungerechtfertigter Bereicherung", "Schadenersatz aus Vertrag",
    "Gewährleistung", "Schadenersatz aus Delikt",
    "offener Werklohnforderung", "Mietzinsrückstand",
    "Darlehensforderung", "Kaufpreisforderung",
]

PFLICHTVERLETZUNGEN = [
    "die vereinbarte Leistung nicht erbracht hat",
    "die Ware mangelhaft geliefert hat",
    "den vereinbarten Termin nicht eingehalten hat",
    "die geschuldete Zahlung nicht geleistet hat",
    "den Vertrag einseitig und ohne Grund aufgelöst hat",
    "fehlerhafte Werkleistungen erbracht hat",
    "die vereinbarten Qualitätsstandards nicht eingehalten hat",
]

EINWENDUNGEN = [
    "die klägerischen Behauptungen unzutreffend sind",
    "die Forderung bereits durch Aufrechnung erloschen ist",
    "der Kläger selbst vertragsbrüchig geworden ist",
    "kein wirksamer Vertrag zustande gekommen ist",
    "die behauptete Pflichtverletzung nicht vorliegt",
    "die geltend gemachte Schadenshöhe nicht nachvollziehbar ist",
]

ENTLASTUNGEN = [
    "er alle ihm zumutbaren Maßnahmen ergriffen hat",
    "höhere Gewalt vorlag",
    "der Kläger die Leistung nicht ordnungsgemäß angenommen hat",
    "der Schaden durch das Verhalten des Klägers selbst verursacht wurde",
    "ein unvorhergesehenes Ereignis die Leistung unmöglich gemacht hat",
]

MITVERSCHULDEN_GRUENDE = [
    "er die Mängel nicht rechtzeitig gerügt hat",
    "er die Schadensminderungspflicht verletzt hat",
    "er die Gefahrenquelle selbst geschaffen hat",
    "er die ihm obliegenden Prüfpflichten unterlassen hat",
    "er trotz Kenntnis der Risiken gehandelt hat",
]

VERJAEHRUNGS_EINREDEN = [
    "der Anspruch bereits verjährt sei (§ 1489 ABGB)",
    "die dreijährige Verjährungsfrist bereits abgelaufen sei",
    "die Forderung jedenfalls präkludiert sei",
    "die kurze Verjährungsfrist des § 933 ABGB greife",
]


def random_date(year_start: int = 2019, year_end: int = 2025) -> str:
    """Zufälliges Datum als String."""
    y = random.randint(year_start, year_end)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{d:02d}.{m:02d}.{y}"


def generate_case() -> dict:
    """Erzeugt einen synthetischen Fall."""
    betrag = round(random.uniform(500, 500_000), 2)
    vertragstyp = random.choice(VERTRAGSTYPEN)
    datum = random_date(2019, 2023)
    mahndatum = random_date(2023, 2025)
    rechtsgrund = random.choice(RECHTSGRUENDE)
    pflichtverletzung = random.choice(PFLICHTVERLETZUNGEN)
    sachverhalt = random.choice(PFLICHTVERLETZUNGEN)
    einwendung = random.choice(EINWENDUNGEN)
    entlastung = random.choice(ENTLASTUNGEN)
    mitverschulden = random.randint(20, 50)
    mitverschulden_grund = random.choice(MITVERSCHULDEN_GRUENDE)
    verjährung_einrede = random.choice(VERJAEHRUNGS_EINREDEN)
    gegenargument = random.choice([
        f"Der Kläger hat den Mangel erst nach {random.randint(3, 12)} Monaten gerügt",
        f"Die behauptete Schadenshöhe von EUR {betrag:,.2f} ist nicht nachvollziehbar belegt",
        "Der Kläger hat die vereinbarten Abnahmefristen selbst nicht eingehalten",
        "Es liegt ein wirksamer Haftungsausschluss vor",
    ])

    # Kläger
    klaeger_template = random.choice(KLAEGER_TEMPLATES)
    klaeger = klaeger_template.format(
        betrag=betrag,
        vertragstyp=vertragstyp,
        datum=datum,
        pflichtverletzung=pflichtverletzung,
        mahndatum=mahndatum,
        rechtsgrund=rechtsgrund,
        sachverhalt=sachverhalt,
    )

    # Beklagter
    beklagter_template = random.choice(BEKLAGTER_TEMPLATES)
    beklagter = beklagter_template.format(
        einwendung=einwendung,
        verjährung_einrede=verjährung_einrede,
        gegenargument=gegenargument,
        entlastung=entlastung,
        mitverschulden=mitverschulden,
        mitverschulden_grund=mitverschulden_grund,
    )

    # Outcome — 3 Klassen mit realistischer Verteilung
    outcome = random.choices(
        ["obsiegen", "teilweise", "unterliegen"],
        weights=[0.45, 0.15, 0.40],
        k=1,
    )[0]

    return {
        "klaegervorbringen": klaeger,
        "beklagtenvorbringen": beklagter,
        "outcome": outcome,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Erzeugt synthetische Testdaten für das LLM-Finetuning"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=100,
        help="Anzahl der zu erzeugenden Fälle (default: 100)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("sample_cases.json"),
        help="Ausgabedatei (default: sample_cases.json)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random-Seed (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    cases = [generate_case() for _ in range(args.count)]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    from collections import Counter
    counts = Counter(c["outcome"] for c in cases)
    print(f"  {len(cases)} Fälle generiert -> {args.output}")
    print(f"  Obsiegen: {counts['obsiegen']} | Teilweise: {counts['teilweise']} | Unterliegen: {counts['unterliegen']}")


if __name__ == "__main__":
    main()
