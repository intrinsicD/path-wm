# Interner Kern: Denken und Vorhersage als ein Kern

28. September 2026. **Grundentscheidungen von Alex getroffen (siehe Ende); nicht implementiert.**
Alex beauftragt, den internen Kern für Denken und Vorhersage/Physik gemeinsam mit
Claude Opus 5.5, Fable 5.1 und Codex gpt-6-astra festzulegen. Fable und Codex haben
unabhängig Entwürfe erstellt; Claude hat sie zusammengeführt, beide haben die
Zusammenführung geprüft. Die Entscheidungen am Ende gehören Alex. Auftrag,
Entwurfszusammenfassungen und Reviewvermerke: `runs/reviews/core_design_20260928/`
(die Volltexte liegen im Sitzungsprotokoll). Mit „Claude“ markierte Punkte sind
Zusammenführungsentscheidungen oder eigene Vorschläge (E11), keine Übereinstimmung beider Entwürfe.

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

## Überraschung und Korrektur (E11)

Vorschlag Claude, von Alex am 28. September angenommen; nicht implementiert.
Überraschung ist die Abweichung zwischen Vorhersage (Prior) und korrigiertem Zustand
(Posterior) pro Slot, etwa KL(Posterior‖Prior) oder die negative Log-Likelihood der
Beobachtung unter dem Prior (größer heißt überraschender); wie Likelihood einzelnen
Slots zugerechnet wird, ist noch zu definieren. Sie wird gegen die übliche Streuung vergleichbarer Slots,
Sichtbarkeit und Ereignisarten normiert. Überraschung ist ein Auslöser für
Prüfungen, kein Urteil, dass Wissen oder Gewichte falsch sind.

- **Grundregeln:** Evidenz geht vor und wird nach Verlässlichkeit gewichtet; eine
  Vorhersage überschreibt nie Evidenz und bleibt als Erwartung nachvollziehbar.
- **Übereinstimmung:** normale Korrektur und Festschreiben der Beobachtung; sie macht
  darauf aufbauende Pläne nicht automatisch verlässlicher. Als Beleg für die Dynamik
  zählt sie erst, wenn die Vorhersage in der Auswertung die Kopierreferenz schlägt; kontrafaktische Beobachtungen prüfen, dass
  die Korrektur der Evidenz folgt und nicht der Vorhersage.
- **Stufen:** gering → normale Korrektur; mittel → Slot als unsicher markieren, beim
  nächsten Mal mehr Detail lesen; groß → Ursachen in fester Reihenfolge prüfen
  (Verlässlichkeit der Beobachtung, Zuordnung/vertauschter Slot, wurde die eigene
  Aktion ausgeführt, fehlende Evidenz; erst danach Verarbeitung) und als Ereignis mit Belegen vermerken.
- **Wiederkehrend:** gleichgerichtete Fehler über mehrere **unabhängige** Instanzen
  erzeugen einen Revisionskandidaten für Wissen (Konzept/Regel); wiederholte Lesungen
  derselben Quelle zählen nicht als unabhängig. Erst spätere Bestätigung macht daraus
  eine Revision; abhängige Vorhersagen und `W`-Inhalte werden invalidiert.
- **Denken:** Widerspricht die Wirklichkeit einem imaginierten Zweig wesentlich (nicht
  nur im Rahmen gewöhnlicher Streuung), werden Plan und
  darauf beruhende Annahmen ungültig; der Agent plant neu.
- **Laufzeit gegen Training:** Zur Laufzeit ändert eine Abweichung nie Gewichte; im
  Training ist der KL-Term ein Lernsignal neben Beobachtungs- und Ergebnisverlusten.
  Vorhandene Bausteine, noch nicht die vorgeschlagene Überraschung pro Slot:
  `ErrorMonitor` (agent.py:132, skalarer Fehlerprädiktor über gemittelte Tokens) und
  Ereignis-Markierung mit Detailbeschreibung (belief.py:537, hybrid_memory.py:337,
  pro Ereignis, nicht pro Slot).

## Lernwege: Gedächtnis zur Laufzeit, Konsolidierung offline (E12)

Alex schlug am 28. September zunächst nullinitialisierte Residual-Adapter vor, die bei
starken Revisionskandidaten zur Laufzeit trainiert, später ins Hauptnetz destilliert
und zurückgesetzt werden. Fable (Kernargumente) und Codex (Bedingungen) rieten unabhängig voneinander davon ab,
Adapter zum Haupt-Lernweg zur Laufzeit zu machen: Das Gedächtnis kann ein Beispiel
sofort behalten und pro Eintrag zurückziehen, nützliche Verallgemeinerung ist damit
nicht garantiert; Gewichtsänderungen brauchen unabhängig geprüfte Belege, und gezieltes
Zurücknehmen einzelner Beispiele ist im Allgemeinen schwerer und nicht garantiert;
jede Aktivierung eines Adapters ist eine neue Modellversion. Gespeicherte Komponenten
tragen ihre Modellversion (pathwm/world_state/store.py:98–125), eine automatische
Verträglichkeitsprüfung oder Invalidierung gibt es dort noch nicht; K/V-Caches siehe
oben „Unsicherheit und Zuständigkeit“. Von Alex angenommener Ablauf:

```
beobachten → Überraschung → Gedächtnis schreiben/revidieren
          → hartnäckige Fehler protokollieren
          → offline Konsolidierung: Training auf protokollierten Belegen + Wiederholung alter Daten
          → neue Modellversion (Checkpoint mit Hash) → Gedächtnis gegen neue Version prüfen → einsetzen
```

- **Zur Laufzeit lernt nur Gedächtnis/Zustand.** Gewichte ändern sich nur in einem
  Konsolidierungslauf, einem gewöhnlichen, von Alex gestarteten Rezeptlauf.
- **Auslöser:** Die Gedächtniskorrektur wurde angewendet, und trotzdem bleibt ein
  gleichgerichteter Fehler über mehrere unabhängige Instanzen; die Prüfung aus E11
  schließt Beobachtung, Zuordnung, Aktion und fehlende Evidenz als Ursache aus.
- **Bedingungen:** Belege mit Herkunft; zurückgehaltene Episoden vor dem Training
  festgelegt; Wiederholung alter Daten; vorab festgelegtes Annahmekriterium (Verbesserung auf
  zurückgehaltenen Fällen **und** keine Verschlechterung auf der unberührten
  Prüfsammlung); neue Version mit Hash; gespeicherte Latente werden gegen die neue
  Version geprüft oder aus Belegen neu kodiert, Caches invalidiert; Zurückrollen stellt
  Modell **und** zugehörigen Gedächtnis-/Latentstand wieder her oder kodiert aus
  Belegen neu; Basis-Checkpoint und fehlgeschlagene Läufe bleiben erhalten.
- **Adapter oder ganzes Netz** ist eine Wahl innerhalb des Konsolidierungslaufs,
  entschieden nach Vergessen und Kosten. Nullinitialisiert heißt: ein Faktor zufällig,
  der andere null, damit das Residual null ist und trotzdem lernen kann.
- **Spätere Option (nicht geplant):** ein zur Laufzeit aktivierter Adapter bei einem
  nachgewiesenen Verarbeitungsfehler; er braucht eine neue Entscheidung von Alex und
  einen eigenen Auswertungsvertrag (Dringlichkeit allein ist keine Ausnahme von E12),
  mit Schattentraining, atomarer Aktivierung und Zurückrollen.
- **Kleinster entscheidender Test (Vorschlag, nicht gelaufen):** in
  `experiments/latent_agent.py` den Kern mit einer eingebauten systematischen
  Verwechslung oder ohne eine Regelfamilie trainieren; zuerst zeigen, dass
  Gedächtniskorrektur den Fehler nicht behebt (eine fehlende Familie muss das
  Gedächtnis nicht überfordern; ein eingebauter Fehler belegt nur diesen Umfang).
  Hauptarm: Adapter mit Wiederholung; Vergleiche mit **gleicher** Wiederholung: ganzes
  Netz, nur Gedächtnis mit gleich vielen zusätzlichen Belegen; Adapter ohne Wiederholung
  als getrennte Ablation (Vergessen). Setzt einen vereinbarten Implementierungsabschnitt
  voraus; eine verkleinerte Vorprüfung müsste auf der vollen Konfiguration wiederholt
  werden.

Quellen (von Fable genannt; Rahmung als komplementäre Lernsysteme von Claude:
McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland 2016), jeweils
nur für die dort untersuchten Methoden und Aufgaben: sequentielles Editieren
verschlechterte die getesteten Modelle ([Gupta et al. 2024](https://arxiv.org/abs/2401.07453));
Kontext schlägt parametrische Editoren auf RippleEdits
([Cohen et al. 2023](https://arxiv.org/abs/2307.12976)); GRACE nutzt ein Codebuch
([Hartvigsen et al. 2023](https://arxiv.org/abs/2211.11031)); LoRA vergaß in den Tests weniger
als volles Feintuning, nicht nichts ([Biderman et al. 2024](https://arxiv.org/abs/2405.09673)).

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
| E11 | Überraschung stufenweise behandeln, Ursachen prüfen, Revisionskandidaten nur aus unabhängigen Fehlern? | ja (angenommen) |
| E12 | Zur Laufzeit lernt nur Gedächtnis; Gewichte nur in Offline-Konsolidierung; Laufzeit-Adapter spätere Option? | ja (angenommen) |

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

- **E11 und E12 angenommen** (später am 28. September): stufenweise Behandlung von
  Überraschung; Gedächtnis als einziger Lernweg zur Laufzeit, Gewichtsänderung nur in
  versionierter Offline-Konsolidierung; Laufzeit-Adapter bleibt spätere Option.

Implementierung ist damit noch nicht begonnen; sie folgt als eigener, mit Alex
geplanter Abschnitt (zuerst Prüfung 1, Verträge).

## Umsetzung E4-A: Kernform im bestehenden `LatentCore` (Plan, 28.09.2026)

Alex: E4 jetzt planen und umsetzen, GPU-Läufe später selbst auf einer anderen Maschine.
**Umfang:** nur die Kernform, nicht der vereinte Kern (Abschnitt B, eigener Plan).

- `LatentCore(..., blocks=1)`: Liste unabhängiger `Block`-Instanzen; jede innere Runde
  wendet die Blöcke nacheinander an (`for r in loops: for b in blocks`), in `induce`,
  `apply` und dem Evidenzleser gleich. `blocks=1` ist bitgleich zum heutigen Verhalten
  (Test); Checkpoint-Schlüssel `block.*` bleiben bei einem Block erhalten.
- Rezeptprofile in `experiments/latent_agent.py`: `full` wird die E4-Kernform
  (Kern Breite 128, 4 Köpfe, 2 Blöcke, 2 Runden; Wahrnehmung unverändert Slots 64,
  Encoder 32); neues Profil `r1` = heutige Form (Breite 64, 1 Block) für Reproduktion
  und als verkleinerte Vorprüfung; `check` bleibt klein. Für die symbolische Stufe
  arbeiten `SymbolicSlots` in Kernbreite; die Pixelstufe erhält eine Projektion 64→128
  vor dem Kern (neu, getestet; eingefrorene Wahrnehmung bleibt eingefroren).
- Kontrollierte Vergleichsachsen als Optionen: `--core-width`, `--core-blocks`,
  `--core-loops` (Kapazitätsregel: Breite {64,128,256}, Tiefe {1,2,4}, Runden {1,2,4,8}).
- Tests (CPU): Parameterzahl 264.832 je Block bei Breite 128, 529.664 für zwei;
  Blockreihenfolge und Rundenzahl; bitgleicher Ein-Block-Standard; Evidenzleser und
  Kontrollarme mit zwei Blöcken; exaktes Fortsetzen; Speicher-Schätzung für 8 GB.
- Übergabe an Alex: Skript mit Worktree-Einfrieren und den Wiederholungen der
  entscheidenden §23-Läufe auf E4 (Einzelfamilien L4 ×2 Seeds, Mischlauf ×2 Seeds,
  Diagnosen aus dem Review: 5×-Batch-Mischlauf, zurückgehaltenes δ). Kein GPU-Lauf durch Claude.

### E4-A umgesetzt (28.09.2026) und Übergabe für Alex' GPU-Läufe

**Umgesetzt (CPU-getestet, 1029+ Tests grün; kein GPU-Lauf durch Claude):**
`LatentCore(blocks=…)` mit Runden über alle Blöcke (`rounds`), `blocks=1` bitgleich und mit
unveränderten Checkpoint-Schlüsseln; Profile `full` = E4 (Kern 128, 2 Blöcke, 2 Runden,
Blöcke 529.664 Parameter, gesamter Kern inkl. Köpfen 733.089) und `r1` = frühere Form;
Optionen `--core-width/--core-blocks/--core-loops`, `--episodes`, `--holdout-ladder-rules`.
Pixelstufe mit breiterem Kern wird mit Hinweis abgelehnt, bis die Projektion 64→128 geplant
und gebaut ist (offen). CI1 (`core_information.py`) nutzt weiter die R1-Form.
**Speicher (Schätzung, CPU-Verhältnis r1→E4 ≈2–2,7× auf M2s-Spitze 2,55 GiB):** ≈5–6,5 GiB
auf der GPU, größter Posten ist die Pool-Auswertung. E10 verlangt 8 GB; daher unten
`--max-reserved-gib 7.5` und ein kurzer Speichertest zuerst.

**Ablauf auf der anderen Maschine** (Stand `main` ab Commit `5dfc912` oder neuer):

```bash
git pull
C=$(git rev-parse --short HEAD); W=../path-wm-frozen/$C
git worktree add --detach "$W" "$C" && cd "$W"
PY=<pfad-zur-venv>/bin/python   # Umgebung wie in pyproject.toml
OUT=<absoluter-pfad>/runs/latent_agent_r1/e4_reruns_$(date +%Y%m%d)
B="--stage symbolic --size full --device cuda --auxiliary-weight 0 --reader evidence --support-sizes 4 8 16 --max-reserved-gib 7.5 --max-minutes 300"
FAM="--train-families category relation open close toggle --train-rules 4 --rule-repeats 7 1 1 1 7 1 1 1 7 1 1 1 7 1 1 1 7 1 1 1"
# 0 Speicher-/Tempotest (Ergebnis: result.json → resources.torch_max_reserved_gib, updates_per_second)
$PY -m experiments.latent_agent $B $FAM --updates 500 --seed 1101 --output $OUT/smoke
# 1 Einzelfamilien (je Seed 1101 und 2202)
for s in 1101 2202; do
  for f in relation category; do $PY -m experiments.latent_agent $B --train-family $f --train-rules 4 --rule-repeats 7 1 1 1 --updates 6000 --seed $s --output $OUT/F_${f}_$s; done
  for f in open close toggle; do $PY -m experiments.latent_agent $B --train-family $f --train-rules 4 --rule-repeats 7 1 1 1 --updates 30000 --seed $s --output $OUT/F_${f}_$s; done
done
# 2 Mischlauf (Referenz M2s, Breite 64)
for s in 1101 2202; do $PY -m experiments.latent_agent $B $FAM --updates 60000 --seed $s --output $OUT/M_$s; done
# 3 Verdünnungstest: 5× Episoden je Update, gleiche Episoden je Familie wie Einzeltraining
for s in 1101 2202; do $PY -m experiments.latent_agent $B $FAM --episodes 80 --updates 12000 --seed $s --output $OUT/M5x_$s; done
# 4 Transfer: δ=3 nie trainiert (Relation und Toggle), Befund im pool_heldout
for s in 1101 2202; do
  $PY -m experiments.latent_agent $B --train-family relation --train-rules 4 --rule-repeats 7 1 1 1 --holdout-ladder-rules 3 --updates 6000 --seed $s --output $OUT/T_relation_$s
  $PY -m experiments.latent_agent $B --train-family toggle --train-rules 4 --rule-repeats 7 1 1 1 --holdout-ladder-rules 3 --updates 30000 --seed $s --output $OUT/T_toggle_$s
done
```

Fortsetzen nach Abbruch: `$PY -m experiments.latent_agent --stage symbolic --resume $OUT/<lauf>` aus
demselben Worktree. Auf einer Maschine mit `systemd-oomd` lange Läufe als Systemdienst mit
`User=` starten (siehe Workflow). **Vorab festgelegt:** Screens wie §23 (je Familie ν_voll ≥ 0,8,
ν_voll−ν_leer ≥ 0,5, ν_voll−ν_permutiert ≥ 0,5); Transfer (4) ist ein Befund mit derselben
Schwelle auf dem zurückgehaltenen Pool, kein Gate. Ein bestandener Lauf gilt als
Entwicklungsbefund; Zuverlässigkeitsaussagen brauchen ≈10 Seeds (Review 28.09.).
Alle Befehlsvarianten wurden mit `--size check` auf der CPU auf gültige Argumente geprüft.
