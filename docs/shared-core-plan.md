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
| S1 | `pathwm/models/core.py`: Zustand, Kern mit fünf Operationen, Schreibrechte, Kontrolloptionen | CPU-Verträge (Prüfung 1) | in Arbeit |
| S2 | Rezept `experiments/core.py`, symbolische Stufe: `induce` aus Belegen, `predict`/`observe` über Druckereignisse, Verluste aus dem Kerndesign | kleiner echter Lauf, Bericht, Vergleich gegen Kopie (Lampe unverändert) und gegen die Kontrollen | offen |
| S3 | Löschen der alten Module, Rezepte und Tests; `WorldSession` und `ConceptMemory` von `belief_state`/`latent_core` lösen | volle Testsuite grün, keine Importe alter Module | offen |
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
