# Gemeinsamer Kern: Umsetzungsplan

29. September 2026. Alex beauftragt, das [Kerndesign](core-design.md) (E1–E12) zu
implementieren, damit auf der neuen Architektur trainiert und getestet werden kann.
Die alte Architektur wird verworfen. Übersicht: [Architekturbild](architecture-overview.html).

## Entscheidungen von Alex (29.09.)

| Frage | Entscheidung |
| --- | --- |
| Alter Stand | Alte Modellmodule samt Rezepten und Tests werden aus `main` gelöscht, **sobald der neue Pfad läuft**. Vorher wird das Tag `archive/pre-core-2026-09-29` gesetzt. Datengeneratoren, Multiskalen-Encoder, Slot-Bausteine, `WorldStore`, Auswertung, Berichte und `runs/` bleiben. |
| Kontrollen aus E2/E3 | Optionen im neuen Kern: `tied=False` (eigene Blöcke je Operation) und `continuous_only=True` (kein kategorialer Code). Keine alten Module als Kontrolle. |
| Wahrnehmung | Wird im neuen Pfad **neu trainiert** (Encoder, Slots, Decoder auf Rule-World-Pixeln), qualifiziert und dann eingefroren (E7). Bis dahin trainiert der Kern auf der symbolischen Stufe. |
| Erste Aufgabe | Rule World: symbolisch zuerst, dann Pixel. |

**Belege auf der anderen Maschine:** `runs/latent_agent_r1/` (E4-Warteschlange,
L4-Leiter, Identitätsprüfung) liegt nicht auf diesem Rechner und ist derzeit nicht
erreichbar. Vergleiche mit alten Ergebnissen nutzen nur die in den Plänen
festgehaltenen Zahlen und sind als nicht nachgeprüft gekennzeichnet. Alte Läufe
lassen sich nach dem Löschen nicht fortsetzen, weil `io.source_record` alle
`pathwm/**/*.py` in die Laufidentität hasht; sie bleiben lesbar.

## Abschnitte

Jeder Abschnitt: Plan → Prüfungen (zuerst rot) → Implementierung mit kleinem echten
Lauf und Bericht → Vergleich. Verkleinerte Vorläufe werden auf der vollen
Konfiguration (Breite 128, 2 Blöcke, 2 Runden, 8 Slots, 4 Szenen-Tokens, 16 `W`,
Code 4×16, K = 4) wiederholt.

| # | Inhalt | Nachweis | Status |
| --- | --- | --- | --- |
| S1 | `pathwm/models/core.py`: Zustand, Kern mit fünf Operationen, Schreibrechte, Kontrolloptionen | CPU-Verträge (Prüfung 1): 15 Tests, volle Konfiguration | erledigt 29.09. |
| S2 | Rezept `experiments/core.py`, symbolische Stufe: `induce` aus Belegen, `predict`/`observe` über Druckereignisse, Verluste aus dem Kerndesign | kleiner echter Lauf, Bericht, Vergleich gegen Kopie (Lampe unverändert) und gegen die Kontrollen | Rezept läuft; Entwicklungsläufe unten; vorab festgelegter Vergleich offen |
| S3 | Löschen der alten Module, Rezepte und Tests; `WorldSession` und `ConceptMemory` von `belief_state`/`latent_core` lösen | volle Testsuite grün, keine Importe alter Module | erledigt 29.09. (siehe unten) |
| S4 | Wahrnehmung neu: Encoder + Slot Attention + Decoder im neuen Rezept, Qualifikation (Bindung, Attribute, Identität, zurückgehaltene Kombinationen), einfrieren | Bericht mit Qualifikationsschwellen, vorab festgelegt | offen |
| S5 | Pixelstufe: Vorhersage/Korrektur auf eingefrorener Wahrnehmung (Prüfung 4) | gegen Kopie und konstante Geschwindigkeit, getrennt nach bewegten/unbewegten Slots | offen |
| S6 | `think` mit Imagination, `apply`; L4-Familien, Kontrollen `tied=False`, `continuous_only` (Prüfungen 5, 6) | vorab festgelegte Schwellen wie §23 | offen |
| S7 | Überraschung je Slot (E11), Laufzeitgedächtnis, Revision/Invalidierung | Lebenslauf mit eingefrorenen Gewichten | offen |
| S8 | Konsolidierungsrezept (E12) | kleinster entscheidender Test aus dem Kerndesign | offen |

## S1: Kern und Verträge

**Schnittstellen.** `CoreConfig` (volle Werte als Standard), `CoreState`
(`h` [B,S,D], `logits` [B,S,4,16] oder `None`, `presence` [B,S], `scene` [B,4,D],
`work` [B,16,D], `revision`, `work_revision`, `hypothetical`), `SharedCore` mit:

- `predict(state, action=None, dt=0, z=None)` → Prior. Ohne Aktion und mit `dt = 0`
  ist das Ergebnis exakt der Eingangszustand (Identität, keine gelernte Näherung).
  Eine Aktion mit `dt = 0` wird genau einmal angewendet. `hypothetical` bleibt erhalten.
- `observe(prior, evidence, valid)` → Posterior-Logits je Slot und
  Zuordnungsvorschlag; `h` kommt aus dem Prior (wie heute). Auf einem
  hypothetischen Zweig verboten. Erhöht `revision`.
- `imagine(state, action, dt, z)` = `predict` auf einer als hypothetisch markierten
  Kopie; das Ergebnis kann nicht beobachtet oder festgeschrieben werden.
- `think(state, goal, memory, branches)` → schreibt nur `W`; liefert Abschluss-Logit,
  Abruf- und Aktionsanfrage. Verzweigungsergebnisse gehen als markierter Kontext ein.
- `induce(evidence, valid)` → Z [B,4,D]; `apply(queries, z)` → Ausgaben, nichts Dauerhaftes.

Alle Operationen laufen durch dieselbe Blockliste (2 Blöcke × 2 Runden, jede Runde
liest denselben Kontext neu). Modus-Embedding und kleine Köpfe unterscheiden sie.
Kein Schrittzähler als Eingabe; Slots haben keine Positions-Embeddings.

**Prüfungen (CPU, volle Konfiguration):** Parameterzahl der zwei Blöcke 529.664;
gebunden teilt dieselben Blockobjekte, ungebunden hat fünf getrennte Stapel;
Slot-Permutation ist äquivariant; Null-Dauer-Identität und Einmal-Anwendung;
Zweig-Isolation (Eingang unverändert, Beobachtung auf Zweig verboten);
Kausalmaske (ungültige Evidenz ändert nichts); Schreibrechte von `think` und `apply`;
Invalidierung von `W` nach Korrektur; exaktes Speichern/Laden des Zustands;
Gradient erreicht die Blöcke aus Prior- und Posterior-Verlust; `continuous_only`.

**Stehende Prinzipien.** *Zustand mit Zuständigkeit und Lebensdauer*: Schreibrechte
und Revisionszähler im Zustand, Zweige als markierte Kopien. *Vorschlagen und Prüfen
trennen*: Imagination schlägt vor, nur `observe` korrigiert. *Rechenaufwand
nützlich verteilen*: feste 2 Runden vor adaptivem Anhalten (zurückgestellt).
*Invarianten wiederverwenden*: K/V-Caching zurückgestellt, bis ein Profil es verlangt.

## S2: Entwicklungsläufe (29.09., Entwicklungsbefund, kein Nachweis)

Volle Konfiguration, RTX 4090 (reserviert 1,4 GiB, also innerhalb der 8-GB-Grenze aus
E10), Seed 1101, 6000 Updates, L4-Relation, Auswertung auf 96 frischen Episoden.
Schwellen waren **nicht** vorab festgelegt; ein Seed. Läufe unter `runs/core/`.

| Lauf | Leser | Einstiegsregel | Tausch | ν Train: voll / leer / getauscht | ν zurückgehalten |
| --- | --- | --- | --- | --- | --- |
| `dev_relation_l4_6k` | Code | nein | 0 | 0,00 / 0,00 / 0,00 | 0,00 |
| `dev_B_code_swap` | Code | 7-1-1-1 | 1 | 0,18 / −0,02 / −0,03 | 0,01 |
| `dev_A_evidence` | Evidenz | 7-1-1-1 | 0 | 0,80 / −0,01 / −0,07 | 0,04 |
| `dev_C_evidence_swap` | Evidenz | 7-1-1-1 | 1 | 0,81 / −0,02 / −0,07 | 0,04 |
| Kopie (Lampe unverändert) | – | – | – | 0,01 | 0,01 |

Ohne Einstiegsregel ignoriert `predict` den Regelkontext völlig (Genauigkeit 0,74 durch
die Abkürzung „kein Treffer“). Mit Evidenz-Leser nutzt der Kern die Belege: ohne oder mit
fremden Belegen fällt ν auf null. Der komprimierte Code Z bleibt der Engpass; auf neue
Regeln wird nicht übertragen. Das deckt sich mit den dokumentierten, hier nicht
nachprüfbaren E4/§23-Befunden. Nächster Schritt: vorab festgelegter Vergleich (Seeds
1101/2202, Schwellen wie §23: ν_voll ≥ 0,8, Abstand zu leer und getauscht ≥ 0,5) mit den
Kontrollen `--untied` und `--continuous-only`.

## S3: Abbau des alten Stands (29.09.)

Gelöscht (50 Dateien, dazu 51 reine Alttest-Dateien): Belief-Agent, Thinker,
`LatentCore`, alte Dynamik- und Gedächtnismodule, Aufgaben-/Abrufköpfe, die alte
Rule-World-Auswertung, die alte `WorldSession` samt `context`/`episodes`/`unified`
und 21 Rezepte, die davon abhingen. Die Sitzung wird in S7 um den neuen Kern neu
gebaut; das Kerndesign verlangt weiter „eine Sitzung, ein Speicher, eine Uhr“.
Bereinigt statt gelöscht: `ConceptMemory` (ohne `ConceptAgent`), `world_state/modules.py`
(ohne `TransitionPredictor`), `explorer.py` (ohne Sitzungs-Laufzeitinspektion),
`understanding.py` (ohne alte Aufgabenauswertung); ν und Kalibrierung liegen jetzt in
`pathwm/evaluation/rules.py`. In gemischten Testdateien wurden nur die Tests auf
Altcode entfernt; der Multiskalen-Test erhält eigene Beispielbilder. Ergebnis: volle
Suite grün, kein Code importiert gelöschte Module. Die Pyramiden-Decoder-Tests
brauchten einen Checkpoint, der nur auf der anderen Maschine liegt; sie gehören zur
alten R1-Wahrnehmung und entstehen in S4 neu.

## S2: vorab festgelegter Vergleich (festgelegt 29.09. vor dem Start, von Alex bestätigt)

**Frage:** Nutzt der neue Kern auf der symbolischen Stufe den Regelkontext, und ändert
sich das mit den Kontrollen aus E2 (ungebunden) und E3 (rein kontinuierlich)?

- **Konfiguration** (aus den Entwicklungsläufen gewählt, das ist offen gelegt): volle
  Größe, L4-Relation, Einstiegsregel 7-1-1-1, `--reader evidence`, `--swap-weight 0`,
  6000 Updates, 16 Episoden je Update, Auswertung auf 96 frischen Episoden (Seed + 7).
- **Arme:** gebunden (Standard), `--untied`, `--continuous-only`; Seeds 1101 und 2202.
  Sechs Läufe unter `runs/core/s2_compare/`.
- **Schwellen je Lauf (wie §23):** ν_voll ≥ 0,8; ν_voll − ν_leer ≥ 0,5;
  ν_voll − ν_getauscht ≥ 0,5, jeweils auf den Trainingsregeln. Ein Arm besteht, wenn
  beide Seeds bestehen.
- **Transfer** auf zurückgehaltene Validierungsregeln wird berichtet, ist aber kein Gate.
- **Aussageumfang:** Entwicklungsbefund mit 2 Seeds; Zuverlässigkeit bräuchte ≈10 Seeds.
  Ein Unterschied zwischen Armen gilt nur als Hinweis, nicht als Nachweis für oder
  gegen das Teilen der Gewichte.

### Ergebnis des festgelegten S2-Vergleichs (29.09.)

Alle sechs Läufe vollständig, Berichte geschrieben, reserviert ≤ 1,61 GiB.

| Arm | Seed 1101: ν voll / leer / getauscht | Seed 2202 | besteht |
| --- | --- | --- | --- |
| gebunden | 0,804 / −0,014 / −0,071 | 0,537 / −0,016 / 0,059 | nein (1/2) |
| `--untied` | 0,158 / 0,008 / −0,020 | 0,219 / −0,017 / 0,054 | nein (0/2) |
| `--continuous-only` | 0,009 (= Kopie) | −0,012 (= Kopie) | nein (0/2) |

Zurückgehaltene Regeln: ν ≤ 0,04 in allen Läufen. **Kein Arm besteht.** Reihenfolge
gebunden > ungebunden > rein kontinuierlich; mit zwei Seeds nur ein Hinweis. Der rein
kontinuierliche Arm bleibt exakt bei der Kopie; das kann an der Lernmechanik liegen
(`h_step` startet bei null und das Ziel der Posterior-Korrektur ist ohne Code schwach)
und ist kein Urteil über kontinuierliche Zustände. Relation L4 mit Seed 2202 verfehlte
auch im alten E4 (0,625, Heimrechner-Beleg). Offen: Z-Engpass, Seed-Abhängigkeit,
Transfer (Aufgabenvielfalt).

## S4: Wahrnehmung neu trainieren (Plan, Schwellen zur Bestätigung durch Alex)

**Aufbau:** `SlotPerception` aus `pathwm/models/slots.py` (Multiskalen-Encoder Breite 32,
Slot Attention Breite 64, Broadcast-Decoder), von Grund auf trainiert als Stufe
`--stage perception` in `experiments/core.py`. Verlust `perception_loss`: RGB-
Rekonstruktion plus Masken-, Rollen-, Attribut- und Lampenlabels, die nur als
Trainingssignal dienen, nie als Agenteneingabe. 8 Slots wie im Kern (7 Entitäten plus
ein freier Slot), sofern `match_slots` das trägt; sonst 7 mit einem abwesenden Kernslot.

**Qualifikation** auf frischen Szenen mit **Validierungs-Maschinentypen** (im Training
nie gesehene Texturen), neue Metrikfunktion `pathwm/evaluation/perception.py`
(Nachbau der alten C1-Messung): Attributgenauigkeit je Attribut, Lampengenauigkeit,
Zeiger auf Maschine und Objekt (der an der Szenenkoordinate gewinnende Slot gehört
zur richtigen Entität), Maschinenerkennung, Rekonstruktionsfehler.

**Vorgeschlagene Schwellen (alte C1-Werte):** Attribut ≥ 0,95 je Attribut, Lampe ≥ 0,99,
Maschinenzeiger ≥ 0,99, Objektzeiger ≥ 0,97; zwei Seeds. Nach Bestehen wird die
Wahrnehmung eingefroren (Hash im Lauf festgehalten) und in S5 verwendet.

**Offen für später** (Literaturscan, Gegenbefund Ramakrishnan et al. 2023): Qualifikation
zusätzlich auf zurückgehaltenen Attributkombinationen; braucht eine Option im Generator.

### S4: festgelegter Lauf (29.09., vor dem Start)

Umsetzung: `--stage perception` in `experiments/core.py`, Metriken in
`pathwm/evaluation/perception.py`, Bilder aus `rw.sample_frames`. Entwicklungs-Smoke
(300 Updates): ≈ 30 Updates/s, 1,74 GiB reserviert; noch nicht qualifiziert (erwartet).

- 40.000 Updates, 32 Bilder je Update, Lernrate 4e-4, volle Größe; Seeds 1101 und 2202.
- Qualifikation auf 1024 frischen Szenen mit Validierungs-Maschinentypen; Schwellen
  wie oben (Attribut ≥ 0,95 je Attribut, Lampe ≥ 0,99, Maschinenzeiger ≥ 0,99,
  Objektzeiger ≥ 0,97). Die Schwellen sind die alten C1-Werte; Alex kann sie ändern,
  bevor S5 darauf aufbaut.
- Eingefroren für S5 wird Seed 1101, falls er besteht, sonst 2202. Besteht keiner,
  ist S4 nicht erfüllt, und die Ursache wird untersucht, bevor S5 beginnt.
