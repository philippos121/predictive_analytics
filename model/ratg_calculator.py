"""
Österreichischer Kostenrechner nach RATG und GGG.

Rechtsgrundlagen:
  RATG  – Bundesgesetz über den Rechtsanwaltstarif
          BGBl. Nr. 189/1999 i.d.F. BGBl. I Nr. 195/2013 und Folgeänderungen
  GGG   – Gerichtsgebührengesetz, BGBl. Nr. 501/1984 i.d.f.
  ZPO   – §§ 40–55 (Kostenersatzrecht):
            § 41  Obsiegensprinzip: Verlierende Partei ersetzt alle Kosten
            § 43  Teilweises Obsiegen: quotaler Kostenersatz bzw. Aufhebung
            § 45  Sofortige Anerkennung / Säumnis

Kostenpositionen im Verfahren:
  - Gerichtsgebühr (GGG TP 1, Pauschalgebühr): bezahlt vom Kläger bei Einbringung
  - Eigene Anwaltskosten: RATG-Tarifposten + Einheitssatz
  - Gegnerische Anwaltskosten: bei Unterliegen gemäß § 41 ZPO zu ersetzen

Hinweis zu den Tarifbeträgen:
  Die RATG- und GGG-Beträge werden periodisch durch BGBl-Kundmachungen
  valorisiert. Die hier verwendeten Werte entsprechen dem Stand ~2024/2025
  und sind ohne Gewähr — bitte gegen das jeweils aktuelle Amtsblatt prüfen.
  Quelle: RATG-Tarifposten-Tabelle, Anlage zum RATG; GGG TP 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ──────────────────────────────────────────────────────────────────────────────
# 1.  RATG-Tarifposten-Tabelle
# ──────────────────────────────────────────────────────────────────────────────
# Streitwert-Obergrenzen (inklusiv) in EUR
_SW_GRENZEN: list[float] = [
    300, 700, 2_000, 3_500, 7_000, 14_000,
    21_000, 35_000, 70_000, 140_000, 280_000, 700_000,
]

# Tarifposten-Werte je Streitwertklasse, in EUR
# Spalten: [TP1, TP2, TP3A, TP3B, TP4, TP5]
#   TP1   Einfache schriftliche Mitteilungen
#   TP2   Einfache Schriftsätze
#   TP3A  Schriftsätze (Klage, Klagebeantwortung) vor BG
#   TP3B  Schriftsätze vor LG, OLG, OGH
#   TP4   Tagsatzungen vor BG
#   TP5   Tagsatzungen vor LG, OLG, OGH
_RATG_TAB: list[list[float]] = [
    #  TP1   TP2    TP3A    TP3B    TP4     TP5
    [    9,   18,    84,    112,    112,    140],   # ≤     300
    [   14,   29,   139,    185,    185,    232],   # ≤     700
    [   22,   44,   204,    272,    272,    340],   # ≤   2 000
    [   31,   63,   300,    400,    400,    500],   # ≤   3 500
    [   44,   88,   407,    542,    542,    678],   # ≤   7 000
    [   64,  128,   596,    794,    794,    993],   # ≤  14 000
    [   87,  174,   808,  1_077,  1_077,  1_347],  # ≤  21 000
    [  116,  233, 1_092,  1_456,  1_456,  1_820],  # ≤  35 000
    [  161,  322, 1_508,  2_011,  2_011,  2_514],  # ≤  70 000
    [  230,  461, 2_158,  2_877,  2_877,  3_596],  # ≤ 140 000
    [  318,  637, 2_984,  3_978,  3_978,  4_973],  # ≤ 280 000
    [  474,  948, 4_441,  5_921,  5_921,  7_401],  # ≤ 700 000
    # > 700 000: Promilleregelung (§ 10 RATG), s. _tp_ueber_700k()
]

_TP_IDX: dict[str, int] = {
    "TP1": 0, "TP2": 1, "TP3A": 2, "TP3B": 3, "TP4": 4, "TP5": 5,
}


def _ratg_zeile(streitwert: float) -> list[float]:
    for grenze, zeile in zip(_SW_GRENZEN, _RATG_TAB):
        if streitwert <= grenze:
            return zeile
    # > 700 000: Promillesätze (§ 10 RATG, Anlage)
    return [
        streitwert * 0.00010,   # TP1
        streitwert * 0.00020,   # TP2
        streitwert * 0.00635,   # TP3A
        streitwert * 0.00846,   # TP3B
        streitwert * 0.00846,   # TP4  (= TP3B)
        streitwert * 0.01057,   # TP5  (= TP3B × 5/4)
    ]


def tp_betrag(tarifpost: str, streitwert: float) -> float:
    """Einzelner RATG-Tarifpostenbetrag in EUR für den gegebenen Streitwert."""
    return _ratg_zeile(streitwert)[_TP_IDX[tarifpost]]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GGG-Pauschalgebühr (Tarifpost 1, erste Instanz)
# ──────────────────────────────────────────────────────────────────────────────
_GGG_GRENZEN: list[float] = [
    150, 300, 700, 2_000, 3_500, 7_000, 14_000,
    21_000, 35_000, 70_000, 140_000, 280_000, 700_000,
]
_GGG_GEBUEHR: list[float] = [
       19,   #  ≤     150
       38,   #  ≤     300
       76,   #  ≤     700
      127,   #  ≤   2 000
      204,   #  ≤   3 500
      340,   #  ≤   7 000
      545,   #  ≤  14 000
      794,   #  ≤  21 000
    1_142,   #  ≤  35 000
    1_977,   #  ≤  70 000
    3_122,   #  ≤ 140 000
    5_624,   #  ≤ 280 000
   10_028,   #  ≤ 700 000
]


def ggg_pauschalgebuehr(streitwert: float) -> float:
    """GGG-Pauschalgebühr für die Klageeinbringung erster Instanz, in EUR."""
    for grenze, gebuehr in zip(_GGG_GRENZEN, _GGG_GEBUEHR):
        if streitwert <= grenze:
            return float(gebuehr)
    return round(streitwert * 0.0144, 2)   # > 700 000: 1,44 %


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Einheitssatz-Faktor (§ 23 RATG)
# ──────────────────────────────────────────────────────────────────────────────
# BG: 50 %, LG / OLG / OGH: 100 %
# (Einheitssatz pauschaliert Aktenstudium, Beratung, einfache Korrespondenz)
_EINHEITSSATZ: dict[str, float] = {
    "BG":  0.50,
    "LG":  1.00,
    "OLG": 1.50,
    "OGH": 1.50,
}


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Typische Verfahrensaufwände (Anzahl Tarifposten pro Instanz/Komplexität)
# ──────────────────────────────────────────────────────────────────────────────
Komplexitaet = Literal["einfach", "mittel", "komplex"]

# Format: {instanz: {komplexitaet: {TP_name: Anzahl}}}
# Annahmen:
#   - Klage + Klagebeantwortung (TP3x × 1 für eigene Seite)
#   - Replik bei "mittel" und "komplex"
#   - Verhandlungstage: 1 / 2–3 / 4–5 (je halber Tag = 1 TP4/5)
#   - TP2: einfachere Schriftsätze (Fristerstreckungen, kurze Eingaben)
_AUFWAND: dict[str, dict[str, dict[str, int]]] = {
    "BG": {
        "einfach": {"TP2": 1, "TP3A": 1, "TP4": 1},
        "mittel":  {"TP2": 1, "TP3A": 2, "TP4": 2},
        "komplex": {"TP2": 2, "TP3A": 2, "TP4": 4},
    },
    "LG": {
        "einfach": {"TP2": 1, "TP3B": 1, "TP5": 2},
        "mittel":  {"TP2": 1, "TP3B": 2, "TP5": 3},
        "komplex": {"TP2": 2, "TP3B": 3, "TP5": 5},
    },
    "OLG": {
        "einfach": {"TP2": 1, "TP3B": 1, "TP5": 1},
        "mittel":  {"TP2": 1, "TP3B": 2, "TP5": 2},
        "komplex": {"TP2": 2, "TP3B": 2, "TP5": 3},
    },
    "OGH": {
        "einfach": {"TP2": 1, "TP3B": 1},
        "mittel":  {"TP2": 1, "TP3B": 2},
        "komplex": {"TP2": 2, "TP3B": 3},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Ergebnis-Datenklasse
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RATGKostenrechnung:
    """
    Ergebnis der RATG/GGG-Kostenberechnung für eine Instanz.
    Alle Beträge in EUR.

    Kostenszenarien (Perspektive des Klägers):

      kosten_bei_obsiegen     = 0
          § 41 ZPO: Gegner ersetzt alle Kosten (Anwalt + GGG) vollständig.

      kosten_bei_teilobsiegen = eigene_anwaltskosten + ggg * 0.5
          § 43 ZPO: Jede Partei trägt eigene Anwaltskosten; GGG-Ersatz
          quotiert nach Obsiegensgrad (Näherung 50 %).

      kosten_bei_unterliegen  = eigene_anwaltskosten + ggg + gegner_anwaltskosten
          § 41 ZPO: Kläger trägt eigene Kosten und ersetzt dem Gegner
          dessen RATG-Kosten; GGG bleibt beim Kläger.
    """
    streitwert:             float
    instanz:                str
    komplexitaet:           str

    # Ausgaben des Klägers
    ggg_pauschalgebuehr:    float = 0.0
    eigene_anwaltskosten:   float = 0.0
    gegner_anwaltskosten:   float = 0.0   # ≈ eigene (gleicher Aufwand)
    einheitssatz_betrag:    float = 0.0
    tp_summe_basis:         float = 0.0

    # Aufschlüsselung der TP-Positionen: {TP_name: {anzahl, einzel, gesamt}}
    tp_positionen:          dict = field(default_factory=dict)

    # Netto-Kostenbelastung je Szenario
    kosten_bei_obsiegen:        float = 0.0
    kosten_bei_teilobsiegen:    float = 0.0
    kosten_bei_unterliegen:     float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Hauptfunktion
# ──────────────────────────────────────────────────────────────────────────────
def berechne_ratg_kosten(
    streitwert: float,
    instanz: str,
    komplexitaet: Komplexitaet = "mittel",
) -> RATGKostenrechnung:
    """
    Berechnet die RATG/GGG-Kosten für ein erstinstanzliches Verfahren.

    Args:
        streitwert:    Klagsbetrag in EUR (> 0).
        instanz:       Gerichtsinstanz: "BG", "LG", "OLG" oder "OGH".
        komplexitaet:  Verfahrensaufwand: "einfach", "mittel" oder "komplex".

    Returns:
        RATGKostenrechnung mit vollständiger Aufschlüsselung und
        den drei ZPO-Kostenszenarien.
    """
    if instanz not in _AUFWAND:
        instanz = "LG"

    aufwand     = _AUFWAND[instanz].get(komplexitaet, _AUFWAND[instanz]["mittel"])
    es_faktor   = _EINHEITSSATZ[instanz]

    # ── TP-Summe berechnen ───────────────────────────────────────────────────
    tp_positionen: dict[str, dict] = {}
    tp_summe = 0.0

    for tp_name, anzahl in aufwand.items():
        einzel  = tp_betrag(tp_name, streitwert)
        gesamt  = einzel * anzahl
        tp_positionen[tp_name] = {
            "anzahl": anzahl,
            "einzel_eur": round(einzel, 2),
            "gesamt_eur": round(gesamt, 2),
        }
        tp_summe += gesamt

    einheitssatz_betrag   = tp_summe * es_faktor
    eigene_anwaltskosten  = tp_summe + einheitssatz_betrag   # = tp_summe × (1 + es_faktor)
    gegner_anwaltskosten  = eigene_anwaltskosten              # gleicher Verfahrensaufwand

    ggg = ggg_pauschalgebuehr(streitwert)

    # ── Kostenszenarien (§§ 41, 43 ZPO) ─────────────────────────────────────
    kosten_bei_obsiegen       = 0.0
    kosten_bei_teilobsiegen   = eigene_anwaltskosten + ggg * 0.5
    kosten_bei_unterliegen    = eigene_anwaltskosten + ggg + gegner_anwaltskosten

    return RATGKostenrechnung(
        streitwert            = streitwert,
        instanz               = instanz,
        komplexitaet          = komplexitaet,
        ggg_pauschalgebuehr   = round(ggg, 2),
        eigene_anwaltskosten  = round(eigene_anwaltskosten, 2),
        gegner_anwaltskosten  = round(gegner_anwaltskosten, 2),
        einheitssatz_betrag   = round(einheitssatz_betrag, 2),
        tp_summe_basis        = round(tp_summe, 2),
        tp_positionen         = tp_positionen,
        kosten_bei_obsiegen       = round(kosten_bei_obsiegen, 2),
        kosten_bei_teilobsiegen   = round(kosten_bei_teilobsiegen, 2),
        kosten_bei_unterliegen    = round(kosten_bei_unterliegen, 2),
    )
