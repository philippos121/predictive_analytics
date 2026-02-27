"""
RATG + GGG Kostenrechner — Rechtsanwaltstarif + Gerichtsgebühren

Schätzt die voraussichtlichen Gesamt-Verfahrenskosten für ein österreichisches
Zivilverfahren in erster Instanz (Bezirks- oder Landesgericht):

  1. Anwaltskosten nach RATG (Rechtsanwaltstarifgesetz BGBl. Nr. 189/1969 idgF)
  2. Gerichtsgebühren nach GGG (Gerichtsgebührengesetz BGBl. Nr. 501/1984 idgF)

Rechtsgrundlagen:
  RATG:
    - Anlage 1 (Tarif): TP-Tabelle nach Streitwert
    - § 23 RATG: Einheitssatz (60 % für SW ≤ 10.170 EUR, 50 % für SW > 10.170 EUR)
    - § 23 Abs 6 RATG: Doppelter Einheitssatz für Klage + Klagebeantwortung
      → Bei SW > 10.170 EUR: ES = 50 % × 2 = 100 % des Tarifansatzes ("100 % ES")
  GGG:
    - TP 1 GGG: Pauschalgebühr für zivilgerichtliche Verfahren 1. Instanz
    - Quelle: BMJ-Richtlinie TP 1–3 (18.06.2024), RIS/GGG
  ZPO:
    - § 41 ZPO: Kostenersatz (Unterlegener trägt die Kosten beider Seiten)

Quellen:
  RATG: https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=10002143
  GGG:  https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=10002667
  BMJ-Richtlinie GGG TP 1–3 (2024):
    https://www.justiz.gv.at/service/gebuehren-und-einbringungsrecht/ggg-richtlinie-tp1-3-und-12-vergleichsgebuehr.e24.de.html

HINWEIS (SCHÄTZUNG):
  TP3A (Haupttarif ZPO) ≈ 3 × TP1, vorb. Schriftsätze ≈ TP1.
  GGG TP1 aus BMJ-Richtlinie 2024 bestätigt für die angegebenen Stufengrenzen.
  Für rechtsverbindliche Beträge: MANZ Tarifrechner (tarif.manz.at).
"""

import math
from typing import Optional


# ─── RATG Anlage 1 — Tarifpost 1 (TP1) ──────────────────────────────────────
# Bestätigt durch RIS/ÖRAK-Recherche (RATG Anlage 1, aktuelle Fassung).
# Format: (streitwert_bis_einschließlich_EUR, tp1_EUR)
_TP1_TABLE = [
    (      40.0,   2.70),
    (      70.0,   3.80),
    (     110.0,   4.90),
    (     180.0,   5.50),
    (     360.0,   6.00),
    (     730.0,   7.30),
    (    1_090.0,  9.70),
    (    1_820.0, 10.60),
    (    3_630.0, 11.90),
    (    5_450.0, 14.20),
    (    7_270.0, 17.60),
    (   10_170.0, 23.30),
    # Über 10.170 EUR: je angefangene weitere 1.450 EUR +2,70 EUR (§ RATG Anlage 1)
]

_ES_GRENZE = 10_170.0   # Einheitssatz-Grenze (§ 23 RATG)
_ES_UNTER  = 0.60       # 60 % bei SW ≤ 10.170 EUR
_ES_UEBER  = 0.50       # 50 % bei SW > 10.170 EUR

# Maximale Stundengebühr für weitere Tagsatzungsstunden (§ RATG Anlage 1 TP3A Abs II)
_MAX_FOLGSTUNDE_EUR = 579.80


# ─── Tarif-Lookup ────────────────────────────────────────────────────────────

def _tp1(streitwert: float) -> float:
    """TP1 aus RATG Anlage 1 (bestätigt). Für SW > 10.170: Formel."""
    for sw_bis, val in _TP1_TABLE:
        if streitwert <= sw_bis:
            return val
    # SW > 10.170 EUR: +2,70 EUR je angefangene 1.450 EUR
    steps = math.ceil((streitwert - 10_170.0) / 1_450.0)
    return round(23.30 + steps * 2.70, 2)


def _tp3a(streitwert: float) -> float:
    """
    TP3A (Haupttarif für ZPO-Verfahren: Klage, KB, Verhandlung).

    TP3A ≈ 3 × TP1 — approximiert aus österreichischer Rechtspraxis.
    Gilt für: Klageschrift, Klagebeantwortung, Tagsatzungen (§ RATG Anlage 1).
    Abweichung zur exakten Tabelle: ±10–15 % (MANZ Tarifrechner für genaue Werte).
    """
    return round(_tp1(streitwert) * 3.0, 2)


def _tp_kurz(streitwert: float) -> float:
    """
    TP für kurze/vorbereitende Schriftsätze (vorb. Schriftsatz, Replik).

    Approximiert als 1 × TP1. Für kürzere Eingaben im ZPO-Verfahren.
    """
    return _tp1(streitwert)


def _es_rate(streitwert: float) -> float:
    """Einheitssatz-Satz gemäß § 23 RATG."""
    return _ES_UNTER if streitwert <= _ES_GRENZE else _ES_UEBER


def _mit_es(tp: float, streitwert: float, es_faktor: float = 1.0) -> float:
    """
    Tarifansatz + Einheitssatz.

    Args:
        tp:          Basis-Tarifansatz (TP-Betrag)
        streitwert:  Fallwert in EUR
        es_faktor:   1.0 = normaler ES, 2.0 = doppelter ES (KB, § 23 Abs 6 RATG)

    Returns:
        Gesamtbetrag netto (ohne USt)
    """
    es = _es_rate(streitwert) * es_faktor
    return round(tp * (1.0 + es), 2)


# ─── Hauptfunktion ───────────────────────────────────────────────────────────

def calculate_ratg_costs(
    streitwert_eur: float,
    klage: bool = True,
    klagebeantwortung: bool = True,
    vorbereitende_schriftsaetze_klaeger: int = 1,
    vorbereitende_schriftsaetze_beklagter: int = 1,
    anzahl_verhandlungen: int = 2,
    stunden_pro_verhandlung: float = 2.0,
    include_ust: bool = True,
) -> dict:
    """
    Schätzt Anwaltskosten nach RATG für beide Parteien.

    Grundstruktur (ZPO-Zivilverfahren, 1. Instanz):
      Kläger:  Klage + vorb. Schriftsätze + Verhandlungen
      Beklagter: Klagebeantwortung + vorb. Schriftsätze + Verhandlungen

    Kostenersatz (§ 41 ZPO):
      - Kläger siegt: bekommt eigene Kosten vom Beklagten erstattet
      - Kläger verliert: zahlt eigene Kosten + Kosten des Beklagten

    Args:
        streitwert_eur:                     Streitwert in EUR
        klage:                              Klageschrift einrechnen
        klagebeantwortung:                  Klagebeantwortung einrechnen
        vorbereitende_schriftsaetze_klaeger:  Anzahl vorb. Schriftsätze Kläger
        vorbereitende_schriftsaetze_beklagter: Anzahl vorb. Schriftsätze Beklagter
        anzahl_verhandlungen:               Anzahl Tagsatzungen gesamt
        stunden_pro_verhandlung:            Ø-Dauer pro Tagsatzung (Stunden)
        include_ust:                        +20 % USt

    Returns:
        dict mit Kostenaufstellung und Gesamtbeträgen
    """
    sw = streitwert_eur
    tp3a_val = _tp3a(sw)
    tp_kurz_val = _tp_kurz(sw)
    es = _es_rate(sw)
    ust = 1.20 if include_ust else 1.0

    # ── Kläger-Kosten ─────────────────────────────────────────────────────────
    k_pos = {}

    if klage:
        # Klage: TP3A + doppelter ES (§ 23 Abs 6 RATG)
        k_pos["klage"] = {
            "bezeichnung": "Klageschrift (TP3A + doppelter ES)",
            "netto": _mit_es(tp3a_val, sw, es_faktor=2.0),
            "tarif_basis": tp3a_val,
            "es_satz": f"{es * 2 * 100:.0f}% (doppelter ES, § 23 Abs 6 RATG)",
        }

    if vorbereitende_schriftsaetze_klaeger > 0:
        einzeln = _mit_es(tp_kurz_val, sw, es_faktor=1.0)
        k_pos["vorb_schriftsaetze"] = {
            "bezeichnung": f"Vorbereitende Schriftsätze ({vorbereitende_schriftsaetze_klaeger}×)",
            "netto": round(einzeln * vorbereitende_schriftsaetze_klaeger, 2),
            "je_netto": einzeln,
            "anzahl": vorbereitende_schriftsaetze_klaeger,
            "tarif_basis": tp_kurz_val,
            "es_satz": f"{es * 100:.0f}%",
        }

    if anzahl_verhandlungen > 0:
        # Tagsatzung: TP3A + normaler ES (erste Stunde), 50 % TP3A für Folgestunden
        verh_liste = []
        for _ in range(anzahl_verhandlungen):
            erste_std = _mit_es(tp3a_val, sw, es_faktor=1.0)
            if stunden_pro_verhandlung <= 1.0:
                kosten = erste_std
            else:
                folge_std = min(
                    _mit_es(tp3a_val * 0.5, sw, es_faktor=1.0),
                    _MAX_FOLGSTUNDE_EUR,
                )
                kosten = erste_std + folge_std * (stunden_pro_verhandlung - 1.0)
            verh_liste.append(round(kosten, 2))

        k_pos["verhandlungen"] = {
            "bezeichnung": f"Verhandlungen / Tagsatzungen ({anzahl_verhandlungen}×, Ø {stunden_pro_verhandlung:.1f}h)",
            "netto": round(sum(verh_liste), 2),
            "je_netto": verh_liste[0] if verh_liste else 0.0,
            "anzahl": anzahl_verhandlungen,
            "tarif_basis": tp3a_val,
            "es_satz": f"{es * 100:.0f}%",
        }

    k_netto = sum(p["netto"] for p in k_pos.values())
    k_brutto = round(k_netto * ust, 2)

    # ── Beklagter-Kosten ──────────────────────────────────────────────────────
    b_pos = {}

    if klagebeantwortung:
        # KB: TP3A + doppelter ES = bei SW > 10.170: 100 % ES (§ 23 Abs 6 RATG)
        b_pos["klagebeantwortung"] = {
            "bezeichnung": f"Klagebeantwortung (TP3A + {es * 2 * 100:.0f}% ES = doppelter ES)",
            "netto": _mit_es(tp3a_val, sw, es_faktor=2.0),
            "tarif_basis": tp3a_val,
            "es_satz": f"{es * 2 * 100:.0f}% (= doppelter ES, § 23 Abs 6 RATG)",
        }

    if vorbereitende_schriftsaetze_beklagter > 0:
        einzeln = _mit_es(tp_kurz_val, sw, es_faktor=1.0)
        b_pos["vorb_schriftsaetze"] = {
            "bezeichnung": f"Vorbereitende Schriftsätze ({vorbereitende_schriftsaetze_beklagter}×)",
            "netto": round(einzeln * vorbereitende_schriftsaetze_beklagter, 2),
            "je_netto": einzeln,
            "anzahl": vorbereitende_schriftsaetze_beklagter,
            "tarif_basis": tp_kurz_val,
            "es_satz": f"{es * 100:.0f}%",
        }

    if anzahl_verhandlungen > 0:
        # Gleiche Verhandlungsgebühren für Beklagten
        verh_liste_b = []
        for _ in range(anzahl_verhandlungen):
            erste_std = _mit_es(tp3a_val, sw, es_faktor=1.0)
            if stunden_pro_verhandlung <= 1.0:
                kosten = erste_std
            else:
                folge_std = min(
                    _mit_es(tp3a_val * 0.5, sw, es_faktor=1.0),
                    _MAX_FOLGSTUNDE_EUR,
                )
                kosten = erste_std + folge_std * (stunden_pro_verhandlung - 1.0)
            verh_liste_b.append(round(kosten, 2))

        b_pos["verhandlungen"] = {
            "bezeichnung": f"Verhandlungen / Tagsatzungen ({anzahl_verhandlungen}×, Ø {stunden_pro_verhandlung:.1f}h)",
            "netto": round(sum(verh_liste_b), 2),
            "je_netto": verh_liste_b[0] if verh_liste_b else 0.0,
            "anzahl": anzahl_verhandlungen,
            "tarif_basis": tp3a_val,
            "es_satz": f"{es * 100:.0f}%",
        }

    b_netto = sum(p["netto"] for p in b_pos.values())
    b_brutto = round(b_netto * ust, 2)

    # ── Gerichtsgebühren (GGG) ────────────────────────────────────────────────
    ggg = calculate_ggg_fees(sw)
    ggg_tp1_eur = ggg["ggg_tp1_eur"]

    # ── Gesamtergebnis ────────────────────────────────────────────────────────
    beide_brutto = round(k_brutto + b_brutto, 2)
    gesamt_mit_ggg = round(beide_brutto + ggg_tp1_eur, 2)

    return {
        # Tarifkennzahlen
        "streitwert_eur": sw,
        "tp1_eur": _tp1(sw),
        "tp3a_eur": tp3a_val,
        "tp_kurz_eur": tp_kurz_val,
        "einheitssatz_rate": es,
        "einheitssatz_pct": f"{es * 100:.0f}%",
        "klagebeantwortung_es_hinweis": (
            f"§ 23 Abs 6 RATG: Doppelter Einheitssatz ({es * 2 * 100:.0f}%) für KB"
        ),
        "include_ust": include_ust,
        # Kläger (RATG)
        "klaeger_positionen": k_pos,
        "klaeger_netto_eur": round(k_netto, 2),
        "klaeger_brutto_eur": k_brutto,
        # Beklagter (RATG)
        "beklagter_positionen": b_pos,
        "beklagter_netto_eur": round(b_netto, 2),
        "beklagter_brutto_eur": b_brutto,
        # RATG Gesamt (relevant bei Niederlage: Kläger zahlt beide Seiten)
        "beide_seiten_ratg_brutto_eur": beide_brutto,
        # GGG Gerichtsgebühren (zahlt Kläger einmalig bei Klageeinbringung)
        "ggg": ggg,
        "ggg_tp1_eur": ggg_tp1_eur,
        # Gesamtbelastung (RATG beider Seiten + GGG) — relevant bei Niederlage
        "gesamt_bei_niederlage_eur": gesamt_mit_ggg,
        "disclaimer": (
            "SCHÄTZUNG nach RATG Anlage 1 (TP3A ≈ 3 × TP1, ±10–15 %) + GGG TP 1 "
            "(BMJ-Richtlinie 2024). Für rechtsverbindliche Beträge: tarif.manz.at. "
            "Quellen: RIS RATG — ris.bka.gv.at (Nr. 10002143); "
            "RIS GGG — ris.bka.gv.at (Nr. 10002667)."
        ),
    }


# ─── GGG — Gerichtsgebühren (TP 1) ──────────────────────────────────────────
# Pauschalgebühr TP 1 GGG für zivilgerichtliche Verfahren 1. Instanz.
# Quelle: BMJ-Richtlinie GGG TP 1–3 (18.06.2024) + RIS GGG
# Format: (streitwert_bis_einschließlich_EUR, pauschalgebühr_EUR)
# HINWEIS: Für SW > 350.000 EUR gilt: 1,2 % × SW + 3.488 EUR (Formel, bestätigt BVwG).
#          Für Zwischenwerte (nicht gelistete Stufen): nächsthöhere Stufe wird verwendet.
_GGG_TP1_TABLE = [
    (     150.0,    25.0),
    (     300.0,    37.0),
    (     700.0,    55.0),
    (   2_000.0,   127.0),
    (   3_500.0,   182.0),
    (   7_000.0,   335.0),
    (  35_000.0,   792.0),
    (  70_000.0, 1_556.0),
    ( 140_000.0, 3_112.0),
    ( 280_000.0, 5_808.0),   # Approximiert (Trendfortschreibung ×2 pro Verdoppelung)
    ( 350_000.0, 7_616.0),   # Approximiert
    # > 350.000: 1,2 % × SW + 3.488 EUR (bestätigt)
]

# TP 2 GGG (Berufung, 2. Instanz) = 50 % von TP 1 (Richtsatz)
_GGG_TP2_FACTOR = 0.50


def ggg_tp1(streitwert: float) -> float:
    """
    GGG Pauschalgebühr TP 1 für Zivilverfahren 1. Instanz.

    Quelle: BMJ-Richtlinie TP 1–3 (2024), RIS GGG.
    Für SW > 350.000: 1,2 % × SW + 3.488 EUR.
    """
    for sw_bis, gebuehr in _GGG_TP1_TABLE:
        if streitwert <= sw_bis:
            return gebuehr
    # Formel für SW > 350.000 EUR
    return round(streitwert * 0.012 + 3_488.0, 2)


def calculate_ggg_fees(
    streitwert_eur: float,
    instanz: str = "LG",
) -> dict:
    """
    Berechnet die GGG-Gerichtsgebühren für ein erstinstanzliches Zivilverfahren.

    Args:
        streitwert_eur:  Streitwert in EUR
        instanz:         "BG" oder "LG" (nur informell; TP 1 gilt für beide)

    Returns:
        dict mit Gebührenaufstellung
    """
    tp1 = ggg_tp1(streitwert_eur)

    return {
        "streitwert_eur": streitwert_eur,
        "ggg_tp1_eur": tp1,
        "ggg_tp1_bezeichnung": "Pauschalgebühr TP 1 GGG (Klage, 1. Instanz)",
        "ggg_hinweis_praet_vergleich": (
            f"Prätorischer Vergleich (§ 433 ZPO) oder Einigung in 1. Verhandlung: "
            f"ermäßigt auf EUR {tp1 / 2:,.2f} (50 % der TP 1)"
        ),
        "ggg_tp2_berufung_eur": round(tp1 * _GGG_TP2_FACTOR, 2),
        "ggg_tp2_bezeichnung": "Pauschalgebühr TP 2 GGG (Berufung, 2. Instanz, ca.)",
        "quelle": (
            "BMJ-Richtlinie GGG TP 1–3 (18.06.2024); "
            "RIS: ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=10002667"
        ),
        "disclaimer": (
            "SCHÄTZUNG. GGG TP 1-Werte aus BMJ-Richtlinie 2024. "
            "Für rechtsverbindliche Beträge: tarif.manz.at oder Justiz.gv.at."
        ),
    }


# ─── EV-Kostenmodell (§ 41 ZPO) ──────────────────────────────────────────────

def compute_cost_risk(
    ratg_result: dict,
    p_win: float,
    p_partial: float,
    p_loss: float,
    partial_success_ratio: float = 0.5,
) -> dict:
    """
    Berechnet die erwartete Kostenbelastung nach § 41 ZPO + GGG.

    Kostenlogik (aus Kläger-Perspektive):
      - Sieg (Vollobsiegen):  RATG-Kosten vollständig erstattet;
                              GGG-Gebühr im Kostenzuspruch inbegriffen → Netto ≈ 0
      - Teilsieg:             Anteiliger RATG-Kostenersatz;
                              GGG anteilig erstattet
      - Niederlage:           Kläger zahlt eigene RATG-Kosten + RATG-Kosten des Beklagten
                              + GGG-Pauschalgebühr (kein Ersatz)

    Args:
        ratg_result:            Ergebnis aus calculate_ratg_costs()
        p_win:                  Wahrscheinlichkeit Vollobsiegen
        p_partial:              Wahrscheinlichkeit Teilobsiegen
        p_loss:                 Wahrscheinlichkeit Unterliegen
        partial_success_ratio:  Anteil Teilerfolg (default 0.5 = 50 %)

    Returns:
        dict mit erwarteter Kostenbelastung (netto, aus Kläger-Sicht)
    """
    k = ratg_result["klaeger_brutto_eur"]
    b = ratg_result["beklagter_brutto_eur"]
    ggg = ratg_result.get("ggg_tp1_eur", 0.0)

    # Szenarien (Kläger-Nettokostenbelastung):
    cost_win     = 0.0                               # Alles erstattet (RATG + GGG)
    cost_partial = (k + ggg) * (1 - partial_success_ratio)  # Anteiliger Eigenanteil
    cost_loss    = k + b + ggg                        # RATG beider Seiten + GGG

    expected_cost = (
        p_win * cost_win
        + p_partial * cost_partial
        + p_loss * cost_loss
    )

    return {
        "erwartete_kostenbelastung_eur": round(expected_cost, 2),
        "kosten_bei_sieg_eur": cost_win,
        "kosten_bei_teilsieg_eur": round(cost_partial, 2),
        "kosten_bei_niederlage_eur": round(cost_loss, 2),
        "klaeger_ratg_brutto": k,
        "beklagter_ratg_brutto": b,
        "ggg_gerichtsgebuehr": ggg,
    }
