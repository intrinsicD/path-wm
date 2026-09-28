# Interner Kern: Denken und Vorhersage als ein Kern

28. September 2026. **Grundentscheidungen von Alex getroffen (siehe Ende); nicht implementiert.**
Alex beauftragt, den internen Kern für Denken und Vorhersage/Physik gemeinsam mit
Claude Opus 5.5, Fable 5.1 und Codex gpt-6-astra festzulegen. Fable und Codex haben
unabhängig Entwürfe erstellt; Claude hat sie zusammengeführt, beide haben die
Zusammenführung geprüft. Die Entscheidungen am Ende gehören Alex. Auftrag,
Entwurfszusammenfassungen und Reviewvermerke: `runs/reviews/core_design_20260928/`
(die Volltexte liegen im Sitzungsprotokoll). Mit „Claude“ markierte Punkte sind
Zusammenführungsentscheidungen, keine Übereinstimmung beider Entwürfe.

## Anforderungen (Alex)

- Ein- und Ausgabe (Encoder, Decoder) schlank; der **interne Kern** hat Leistung:
  nicht so viel wie möglich, sondern so viel wie nötig, im Zweifel ausreichend.
- Denken und Vorhersage lernen beide, wie sich Codes im latenten Raum verändern
  (durch Zeit und Handlung beziehungsweise durch Nachdenken).
- Minimale externe Abhängigkeiten; das [Forschungsziel](integrated-latent-agent-goal.md)
  bleibt maßgeblich: kompatible latente Rechnung, geteilte Tiefe durch Schleifen,
  Konzept-/Instanzgedächtnis mit Korrektur ohne Nachtraining.

## Heutiger Stand (Code geprüft)

| Teil | Ort | Aufbau |
| --- | --- | --- |
| `Thinker` | [agent.py:105](../pathwm/models/agent.py), `think()` ab :709 | ein Attend-Baustein, `steps`-mal mit geteilten Gewichten; ändert nur 8 Arbeits-/Reasoning-Tokens; Rückmeldung enthält den rohen Schrittzähler (:741) |
| `LatentDynamics` | agent.py:71 | 2 Attend-Blöcke, `x + delta·dt`, Gaussian-Pfad |
| `BeliefDynamics`, `BeliefCorrection` | [belief.py:33, :74](../pathwm/models/belief.py) | Prior/Posterior 8×8 Kategorien aus **einem gemittelten Vektor**; `dt` als Eingabe, bewusst ohne Multiplikation (:64); Standardpfad der `WorldSession` |
| `LatentCore` (R1) | [latent_core.py:46](../pathwm/models/latent_core.py) | ein gebundener Block (Cross-, Self-Attention, FF; Breite 64, 4 Köpfe, 2 Schleifen; 66.880 Parameter) für Konzeptbildung `G` und Anwendung `T` |
| `Predictor` | temporal.py:43 | Merkmalsgitter-Vorhersage mit GRU-Gedächtnis |
| `TransitionPredictor` | world_state/modules.py:275 | MLP-Vorhersage mit Streuung |

Sieben Teilmodule mit verschiedenen Tokenformaten. Denken und Imagination sind über
den Task-Dispatch nur einseitig verbunden (agent.py:688–703): Ein Imaginationszweig
geht an den Aufrufer zurück, nicht in eine weitere Denkrunde. Mean-Pooling begrenzt
den kategorialen Kopf; dass dadurch Objektinformation im kontinuierlichen Zustand
fehlt, ist nicht gezeigt. Dünner Thinker und Schrittzähler sind Risiken, keine
gezeigten Fehler. Belege: Kurze Bildvorhersagen schlagen die Kopierreferenz noch
nicht ([Projektstand](project-state.md)); mehr Denkschleifen brachten im
Auslesevergleich keinen Nutzen ([latenter Kern](latent-core.md)); der gesampelte
Richtungscode liegt nahe 50 % ([Auslese-Plan](modality-readout-plan.md), :357, :387).

## Übereinstimmung von Fable und Codex

1. **Gebundene Berechnung** nach dem Muster des `LatentCore`-Blocks (Cross-, Self-
   Attention, FF), wiederholt über Schleifen **und** geteilt zwischen allen
   Operationen. Operationen unterscheiden sich durch gelernte Modus-Embeddings,
   kleine Eingangsprojektionen und getrennte Köpfe; Prior und Posterior teilen
   keinen Kopf. Die Blockzahl ist ein Unterschied (siehe unten).
2. **Objektbezogener Zustand:** Entitäts-Slots mit kontinuierlichem `h` **und**
   eigenem kategorialem Code pro Slot, wenige Szenen-/Global-Tokens, ein
   Arbeitsbereich `W` und abgeleitete Konzeptcodes `Z` (K = 4 Tokens pro Konzept).
   Das ersetzt den einen gemittelten Weltcode. Slot-Index ist keine Identität;
   Zuordnung bleibt vorschlagbar, unsicher und offen, nie erzwungen.
3. **Schreibrechte getrennt** (Tabelle unten); Imaginationszweige sind Kopien und nie
   festschreibbar (vorhanden: belief.py:532–533).
4. **Imagination im Denken:** Eine Denkrunde kann denselben Vorhersageoperator auf
   einem Zweig laufen lassen; das Ergebnis kommt als markierter Kontext zurück.
5. **Kein roher Schrittzähler** als Eingabe; der rekurrente Zustand trägt den Fortschritt.
6. **Migration:** neues, optionales Modul mit Adaptern auf die vorhandenen
   Signaturen (agent.py:41, :80, :113); alte Module, Checkpoints, Rezepte und Läufe
   bleiben Vergleichsbasis. Alte Checkpoints gelten nicht als kompatibel, nur weil
   Tensorformen passen.
7. **Kapazität nach Messung** statt nach Vorgabe (Regel unten).

## Operationen und Schreibrechte

| Operation | Abfragen | Kontext | Schreibt | Kopf |
| --- | --- | --- | --- | --- |
| `observe` (Korrektur) | Prior von Slots und Szene | Encoder-Tokens, Gedächtnislesung, Aktion/Zeit-Token | Posterior-Zustand; nur ein beobachtetes Ereignis schreibt Evidenz fest | Posterior-Logits pro Slot, Zuordnungsvorschlag |
| `predict(dt, Aktion)` | Slots, Szene | Slots, Szene, Aktions-Token mit Präsenzflag, `dt`, `Z` | erschlossenen (nicht beobachteten) Zustand; bei Imagination nur den Zweig | Prior-Logits pro Slot, Residual für `h` |
| `think` | `W` | Slots, Szene, `Z`, Gedächtnis, Ziel, markierte Zweigergebnisse | nur `W` und Vorschläge | Aktion, Abrufanfrage, Abschluss |
| `induce` | Konzept-Seeds | gültige Belege | abgeleitete `Z` (aus Belegen neu berechenbar) | unverändert latent_core.py:68–80 |
| `apply` | Slot-/Rollen-Abfragen | `Z` | nichts Dauerhaftes | Einordnung, Beziehung, hypothetischer Übergang |

Revisionen aus Schlussfolgerungen laufen nur über einen ausdrücklichen Revisionsweg
mit Abhängigkeiten ([Entity-Memory-Design](entity-memory-design.md)); Gedanken werden
keine Beobachtungen. Nach Korrekturen werden abhängige Vorhersagen und `W`-Inhalte
invalidiert.

## Zeit und Ereignisse (Claude-Zusammenführung)

Dauer null ohne Ereignis ist die Identität. Ein augenblickliches Ereignis (Knopfdruck)
wird genau einmal angewendet; anhaltende Steuerung (Schieben über eine Dauer) ist
davon getrennt. Danach entwickelt sich der Zustand über `dt` als Eingabe; lange
Horizonte wiederholen einen Basisschritt. Codex ließ auch dauerabhängige
Residualraten mit begrenzten Teilschritten zu; das bleibt Vergleichsvariante.
Zerlegungskonsistenz (Δ₁ + Δ₂ gegen Δ) wird geprüft (Claude); exakte Physik wird
nicht versprochen.

## Schleifen

- **Innen:** Der Block wird pro Operation R-mal angewendet; in jeder Runde liest er
  denselben unveränderlichen Eingabekontext erneut per Cross-Attention (so arbeitet
  der `LatentCore`-Block bereits). Zusätzliches additives Wiedereinspeisen des
  Startzustands ist eine Vergleichsvariante (Geiping et al. 2025).
- **Außen:** Denkzyklen wählen Abruf, Simulation oder Antwort. Physikalische Zeit,
  innere Runden und äußere Zyklen sind getrennte Größen.
- **Reihenfolge (Claude):** feste R zuerst, wie in der
  [Compute-Bewertung](compact-compute-assessment.md); zufälliges R erst nach
  nützlichem Verhalten bei fester Tiefe. Tests mit mehr Schleifen als im Training
  sind erst danach aussagekräftig.
- **Zwischenverluste (Hypothese):** Verlust auch nach Zwischenrunden mit normiertem
  Gewichtungsplan, gegen eine Kontrolle nur mit Endverlust; gleich starker Druck in
  jeder Runde kann nützliche Zwischenrechnung auch behindern. In Vergleichen
  „gebunden gegen ungebunden“ gilt für beide Arme dasselbe Ziel.

## Lernsignale

- **Vorhersage:** Posterior-Rekonstruktion plus reine Prior-Mehrschrittvorhersage.
  KL-Koeffizienten aus dem [Belief-Rezept](belief-model.md) übernommen (Claude):
  Dynamik/Repräsentation 1/0,1 mit Stop-Gradient, 1 % Unimix. Das freie Nat wird dort
  **nach dem Summieren der Gruppen pro Ereignis** angewendet; pro Slot wäre es bei
  8 Slots achtmal so viel Spielraum. Reduktion über Gruppen, aktive Slots und Batch
  ist eine offene Entscheidung (E9).
- **Gegen Kollaps:** Konsistenz gegen `sg(posterior)` allein genügt nicht, wenn
  Repräsentation und Vorhersage gemeinsam lernen. Zuerst eingefrorene, qualifizierte
  Wahrnehmung als Ziel; beobachtbare Ergebnisse, Rekonstruktion und kontrafaktische
  Empfindlichkeit; Code-Auslastung und Decoder-Umgehung überwachen.
- **Denken:** Aufgaben-/Ergebnisverluste durch die entrollten Schleifen, geprüfte
  Rangfolgen von Aktionen; keine erfundenen Denkspuren belohnen.
- **Konzepte:** vorhandenes `episode_loss` (latent_core.py:187); bekannter Fehler
  „`T` ignoriert `Z`“ wird mit einem Vertauschungsterm (Fable) adressiert.
- **Reihenfolge:** Vorhersage/Korrektur zuerst mit eingefrorener Wahrnehmung;
  gemeinsames Training mit Denkzweigen beginnt, wenn Prüfung 4 besteht. Bis dahin
  ist das Prinzip „das System trainieren, das laufen wird“ bewusst noch nicht erfüllt.

## Unsicherheit und Zuständigkeit

Maßgeblich sind Rohquellen, ausgeführte Aktionen mit Ergebnissen, Ereigniszeiten
und versionierte Zuordnungen. Beliefs und Konzeptcodes sind abgeleitete Schätzungen;
der exakte rekurrente Zustand wird für Neustart gespeichert und als Inferenz
gekennzeichnet. `W` bleibt innerhalb einer Aufgabe erhalten, wird bei exaktem
Fortsetzen wiederhergestellt und an einer neuen Aufgabe zurückgesetzt. Kategoriale
Entropie ist keine kalibrierte Sicherheit: Beobachtbare Ergebnisverteilungen werden
auf Kalibrierung geprüft; Identitätszuordnung trägt eigene Wahrscheinlichkeiten.
Zwischengespeicherte K/V sind nur gültig bei gleicher Quelle, gleichen Gewichten,
gleicher Repräsentations-/Zustandsidentität, Masken und Zeitmetadaten; im Training
bleiben Gradienten erhalten.

## Unterschiede und Empfehlung

| Frage | Fable | Codex | Empfehlung (Claude) |
| --- | --- | --- | --- |
| Breite | 64 | 128 | **128** als volle Konfiguration („im Zweifel ausreichend“); 64 nur als verkleinerter Erstlauf |
| Blöcke | 1 | 2 verschiedene, gemeinsam wiederholt | **2**; ob Tiefe 2 nötig ist, klärt die Leiterachse Stapeltiefe {1, 2, 4} |
| Köpfe bei 128 | – | 4 | **4** (Kopfbreite 32 wie heute) |
| Slots / `W` / Szene | 8 / 8 / 2 | 16 / 16 / 4 | **8** Slots mit Präsenz-Logit (Claude; R1 nutzt 7), **16** `W`, **4** Szenen-Tokens |
| Code pro Slot | 4×8 | 4×16 | **4×16**: wenig Kosten, Reserve gegen Code-Kollisionen |
| Innen/außen, Evidenzbudget | R zufällig, Mittel 4 | 2 innere Runden, äußerer Zyklus, 128 gelesene Evidenz-Tokens | **2 innere Runden fest**, äußerer Zyklus, **128** gelesene Evidenz-Tokens |
| Simulation im Denken | 1 Zweig, Tiefe ≤2 | ≤4 Zweige, Horizont 4 | **1 Zweig, Tiefe 2** als Einstieg; Zielkonfiguration 4×4 |
| Training | gemeinsam mit Tiefenaufsicht | Vorhersage zuerst | **Vorhersage zuerst**, gemeinsam als Kontrolle |

Größe: Zwei Blöcke bei Breite 128 haben **529.664 Parameter** (ohne Adapter und
Köpfe; von beiden geprüft). Zum Vergleich: Das ganze heutige Belief-Modell
einschließlich Encoder und Decoder hat 515.553. Die Wahrnehmung (Slots Breite 64,
Encoder Breite 32) wird über eine Projektion angeschlossen und beim ersten
Kerntraining eingefroren.

## Kapazitätsregel („so viel wie nötig, im Zweifel ausreichend“)

Achsen: Breite {64, 128, 256}, Stapeltiefe {1, 2, 4}, innere Runden {1, 2, 4, 8},
Slots {4, 8, 16}. Erst prüfen, ob die nötige Information den Kern erreicht; dann eine
Achse nach der anderen, die mit dem größten Gewinn pro zusätzlicher Rechnung zuerst.
Die kleinste Konfiguration, die die vereinbarten Schwellen erreicht, wird gewählt.
**Im Zweifel**, also bei weiterhin ungeklärter Unteranpassung, wird die größere
Variante innerhalb der VRAM-/Latenzgrenze behalten. Berichtet werden Qualität gegen
Trainingszeit und Schritte, Parameter, Optimierer-/Aktivierungsspeicher, gelesene
gegen gehaltene Tokens und Latenz. Encoder und Decoder bleiben schlank, solange
eingefrorene Proben den Engpass nicht dort zeigen; Multiskalen-Details bleiben
abrufbar.

## Prüfungen in Reihenfolge

Jede verkleinerte Vorprüfung (Breite, Slots, Runden, Simulationsbudget) muss danach
auf der vollen Konfiguration wiederholt werden; bis dahin gilt die Prüfung auf dem
vollen Modell als unvollständig. Schwellen, Seeds und Budgets werden vor jedem Lauf
festgelegt.

1. **Verträge** (CPU): Zweig-Isolation, Kausalmasken, Null-Dauer/Ereignis-Semantik,
   Slot-Permutation, Neustart, Invalidierung. Der bestehende
   `--state-model belief --check` sichert nur die alten Pfade; der neue Pfad braucht
   eigene Prüfungen. Rezept: `experiments/multimodal.py`.
2. **Schrittkodierung:** heutiger Thinker gegen „ohne Schrittzähler“ bei festen
   1/2/4 Runden; Rezept: `experiments/modality_readout.py`.
3. **Slot-Codes gegen gemittelten Code** auf der Richtungsaufgabe
   (`runs/modality_repair_v1`): vorab festgelegte Verbesserung gegenüber passenden
   Kontrollen, nicht nur „über 50 %“.
4. **Dynamik:** vereinter Vorhersageoperator gegen Kopierreferenz, 1 und 4 Schritte,
   verdeckte und kreuzende Objekte, ungesehene Dauern; Rezept: `experiments/dynamics.py`.
5. **Teilen trennen:** (a) Teilen über Operationen und (b) Teilen über Tiefe
   getrennt prüfen, bei gleichem 2er-Stapel und gleicher ausgeführter Tiefe; L4-Familien,
   2 Seeds (bekanntes Risiko: saatabhängiger Einstieg beim gemischten Kern);
   Rezept: `experiments/latent_agent.py`.
6. **Imagination im Denken** auf der Zwei-Tasten-Kette gegen direkte Anwendung und
   gegen gleich viel Denken ohne Simulation; Rezept: `experiments/latent_agent.py`.
7. **Kapazitätsleiter** erst nach 4–6; danach Lebenslauf mit eingefrorenen Gewichten:
   Konzept erwerben, anwenden, planen, Widerspruch erhalten, abhängige Antworten korrigieren.

## Stehende Prinzipien

*Zustand mit Zuständigkeit und Lebensdauer* (Schreibrechte, wegwerfbare Zweige,
aufgabengebundenes `W`); *Vorschlagen und Prüfen trennen* (Imagination schlägt vor,
nur Beobachtung legt Evidenz fest); *Invarianten wiederverwenden* (Belege einmal
vorbereiten, K/V nur bei voller Identität); *Rechenaufwand nützlich verteilen*
(feste Budgets vor adaptivem Anhalten); *das System trainieren, das laufen wird*
(ab Prüfung 4 Zweige und Schleifen im Training). Zurückgestellt: adaptives Anhalten,
Experten-Routing, latente Diffusion, verschachtelte Zwei-Frequenz-Schleifen (HRM)
als gekennzeichnete Option. Erwarteter Nutzen: ein Kern statt sieben Teilmodulen,
dichtes Lernsignal für den Denkoperator durch geteilte Vorhersage. Kosten:
Kopplungsrisiko (Verdünnung zwischen Aufgaben), mehr Aktivierungsspeicher durch
entrollte Schleifen.

## Quellen (von Fable/Codex genannt)

Universal Transformers ([1807.03819](https://arxiv.org/abs/1807.03819)); Slot Attention
([2006.15055](https://arxiv.org/abs/2006.15055)); SlotFormer ([2210.05861](https://arxiv.org/abs/2210.05861));
DreamerV3 ([2301.04104](https://arxiv.org/abs/2301.04104)); Geiping et al. 2025, Tiefe durch
Rekurrenz ([2502.05171](https://arxiv.org/abs/2502.05171)); HRM ([2506.21734](https://arxiv.org/abs/2506.21734));
TRM ([2510.04871](https://arxiv.org/abs/2510.04871)); Coconut ([2412.06769](https://arxiv.org/abs/2412.06769));
V-JEPA ([2404.08471](https://arxiv.org/abs/2404.08471)); TD-MPC2 ([2310.16828](https://arxiv.org/abs/2310.16828)).
Keine Quelle belegt diese Kombination.

## Entscheidungen für Alex

| # | Frage | Empfehlung |
| --- | --- | --- |
| E1 | Objekt-Slots plus Szenen-Tokens statt unstrukturierter Welt-Tokens? | ja |
| E2 | Ein geteilter Kern für Denken und Vorhersage (getrennte Köpfe) statt getrennter Netze? | ja, mit ungebundener Kontrolle |
| E3 | Kontinuierlich plus kategorial pro Slot? | ja, rein kontinuierlich als Kontrolle |
| E4 | Volle Konfiguration: Breite 128, 2 Blöcke, 4 Köpfe, 8 Slots, 4 Szenen-Tokens, 16 `W`, 4×16 Codes, 2 innere Runden, 128 Evidenz-Tokens? | ja |
| E5 | Kein Schrittzähler als Eingabe? | ja |
| E6 | Ereignisse diskret zuerst (R1-Semantik), kontinuierliche Zeit getrennt validieren? | ja |
| E7 | Vorhersage zuerst mit eingefrorener Wahrnehmung, gemeinsames Training ab Prüfung 4? | ja |
| E8 | Simulations-Zielkonfiguration 4 Zweige × Horizont 4 (Einstieg 1×2)? | ja |
| E9 | Freies Nat pro Ereignis (wie heute) oder pro Slot? | pro Ereignis, pro Slot als Vergleich |
| E10 | Welche VRAM-/Latenzgrenze und welche Annahmeschwellen gelten? | offen, von Alex festzulegen |

### Entscheidung Alex, 28. September 2026

- **E1–E3, E5 angenommen:** ein gemeinsamer gebundener Kern für Denken und Vorhersage
  mit getrennten Köpfen; Objekt-Slots plus Szenen-Tokens und Arbeitsbereich;
  kontinuierlich plus kategorial pro Slot; kein Schrittzähler als Eingabe. Getrennte
  Netze und rein kontinuierlicher Zustand bleiben Kontrollen.
- **E4 angenommen:** volle Konfiguration Breite 128, 2 Blöcke (529.664 Parameter im
  Kern) mit den übrigen Werten aus E4; Breite 64 nur als verkleinerter Erstlauf mit
  Pflicht-Wiederholung auf der vollen Konfiguration.
- **E6–E7 angenommen:** Vorhersage zuerst mit diskreten R1-Ereignissen und
  eingefrorener Wahrnehmung; Denken mit Simulation kommt hinzu, sobald Prüfung 4
  besteht; gemeinsames Training als Kontrolle.
- **E10 Ressourcengrenze:** Training **und** Inferenz müssen lokal in **8 GB** passen
  (RTX 3050). Das begrenzt die Kapazitätsleiter nach oben.
- **Nicht ausdrücklich entschieden:** E8 (Simulationsziel 4×4) und E9 (freies Nat pro
  Ereignis) gelten als Arbeitsannahme gemäß Empfehlung; Annahmeschwellen werden vor
  jedem Lauf mit Alex festgelegt.

Implementierung ist damit noch nicht begonnen; sie folgt als eigener, mit Alex
geplanter Abschnitt (zuerst Prüfung 1, Verträge).
