# Video understanding: proposed evaluation map

16 September 2026. Requested capability overview, not an adopted run protocol.
No new training, benchmark download or validation promotion. Numerical gates,
populations and budgets must be declared before any experiment using this map.

## Objective and present evidence

For PATH-WM, video should provide persistent, queryable and predictive information
about entities, changes and interactions to the shared latent core and World State.
Direction is one genuine component of this objective. Coverage of a constructed
direction task does not establish the other components or natural-video transfer.

The shared image/video codec has tested mechanics. Reconstruction studies do not
establish understanding. The latest fixed correspondence rule and structured reader
pass a narrow fresh controlled-pan screen, but learned refinement fails source
preservation. See [results](video-evidence-plan.md). Natural object trajectories,
streaming memory and useful integration of this codec into the agent remain open.

## Capability matrix

| Capability | Concrete task | Measurement |
| --- | --- | --- |
| Visual state and detail | Locate objects and distinguish task-relevant properties, including small objects | Localization/property accuracy by object size and resolution |
| Motion | Stillness, horizontal/vertical/diagonal motion, rotation, speed and acceleration | Direction/stillness accuracy and displacement/velocity error with explicit coordinate and time units |
| Camera versus object motion | Moving camera/static object, static camera/moving object and both moving | Foreground/background motion error; correct ambiguity handling where separation is unobservable |
| Identity and tracking | Similar objects cross, leave view, become occluded and return | Track position, identity switches, duplicate/merge errors and reacquisition performance; point tracking alone does not prove semantic identity |
| State and relation changes | Open/close, empty/fill, pick up/put down, inside/outside and contact | Per-entity state/relation correctness and transition timing |
| Interactions and event order | Who moved which object, before/after, repeated actions and multi-step events | Actor/object binding, temporal localization, order and count accuracy |
| Persistent history | Same final image after different relevant histories; delayed queries beyond recent context | Correct historical answers and entity binding versus memory delay/distractors |
| Prediction | Predict future positions, states, contacts and events from prefixes only | Error by horizon and appropriate distribution scores versus persistence and constant-velocity references; forecast uncertainty |
| Action effects and causal alternatives | Same initial condition with different recorded actions or controlled interventions | Predicted outcome and task success; observational video alone is not a ground-truth causal test |
| Uncertainty | Occluded, noisy, ambiguous or unseen cases | Error versus declared confidence, coverage versus error when abstaining, correction after new evidence |
| Shared-core and World State use | Feed video evidence through adapter, binding, update and retrieval, then query/predict/act | End-task correctness and benefit over matched no-history/shuffled-history controls, plus stage diagnostics |
| Multimodal correspondence | Associate visual events with sounds or instructions, including mismatches and missing modalities | Synchronization, association and grounded-answer accuracy; report video-only and audio-only controls |

## Checks across all capability tests

- Identify targets, coordinate conventions, timestamps and annotation quality. Tiny
  crops can remove decisive evidence; compare supported resolutions rather than
  imposing 48x48 as the capability contract. Avoid claims about metric 3D velocity
  without the required calibration/observability.
- Use single-frame and unordered/shuffled-history controls where appropriate. Build
  matched-history cases needing different answers. A temporal benefit must depend
  on the relevant sequence, not merely extra scene appearance. Reversing video is
  useful only for tasks whose target actually transforms that way.
- Hold out people/objects/scenes/source recordings as appropriate; near-duplicate
  clips cannot cross splits. Separate development from untouched confirmation,
  including previously inspected sources as regression checks.
- Stress different speeds, lighting, blur, compression, viewpoints, occlusions,
  frame rates, dropped frames, scene cuts, clip lengths and distractors. Report
  per-group failures and several seeds; account for source-level sample dependence.
- For online use, test no future leakage, state across chunks, explicit reset,
  episode isolation, missing/late inputs, bounded memory, latency and GPU memory.
  Existing finite-window causal checks do not validate streaming state.
- Fix readout capacity/training exposure for feature probes. Read comparable targets
  from frame features, temporal features, core state and retrieved memory. A failed
  probe is not proof of erased information; a successful probe is not proof that
  the deployed agent uses it. Test the actual downstream path too.
- Record performance versus training data and compute, plus preservation of image
  capabilities. Reconstruction and generation quality have separate reports; neither
  is a substitute for state/behavior tests. Understanding need not be tested solely
  through a language decoder.

## Proposed sequence and stopping discipline

1. Preserve controlled pans as a diagnostic/regression test. Establish natural
   object motion and tracking first, including stillness, variable speed, camera
   motion and short occlusion, with independent temporal references.
2. Feed that evidence through the actual shared core and entity memory; test
   persistent identity, state changes and history-dependent queries.
3. Add future state/event prediction at multiple horizons; require relevant-history
   benefit over matched simple references before attributing success to dynamics.
4. Add action-conditioned tasks, longer events and audiovisual interaction as their
   relevant interfaces become testable. These need their own scope and gates.

This sequence is proposed, not authorization to execute all stages. Each bounded
milestone needs a fixed primary endpoint, source split, minimum meaningful effect,
preservation limits, resource budget and stop rule before running. Passing advances
that scope; failure permits a targeted cause-localizing diagnosis with a declared
budget, not an automatic chain of new architecture variants. Preserve negative
results and revisit the overall bottleneck before extending the experiment series.

## External reference tasks

- [TAP-Vid](https://tapvid.github.io/) supplies real and synthetic annotated point
  tracks for trajectory/occlusion evaluation; it is not a full entity-memory test.
- [Something-Something v2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
  supplies videos of object interactions for fine-grained action tasks.
- [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)
  evaluates understanding, anticipation and action-conditioned planning separately.

These are methodology references, not downloaded data, adopted architectures or
capabilities already measured in PATH-WM. Existing Charades clips can support a
small initial natural-video study, but their available annotations must first be
checked against the desired trajectory/state targets.

## Bild-/Videoverstehen nach der Personen- und Engine-Diskussion

22. September: Alex fragt, welche weiteren Fähigkeiten das Modell für echtes
Bild-/Videoverstehen benötigt. Die Personenrekonstruktion liefert spezialisierte
Beobachtungen; die [Neural Engine](neural-engine-inference.md) organisiert deren
Ausführung. Beides allein belegt noch kein Verstehen. Operationaler Vorschlag:
Das System kann aus der jeweils verfügbaren Evidenz richtige Objekt-/Relations-
und Ereignisurteile bilden, relevante Verläufe behalten, neue Kombinationen
verarbeiten und bei fehlender Evidenz Unsicherheit ausdrücken. Ein anschauliches
Mesh, niedriger Rekonstruktionsfehler oder flüssige Bildbeschreibung reichen als
Prüfung nicht aus; Sprachproduktion ist keine Voraussetzung jeder visuellen Aufgabe.

Zusätzlich zur Personenerfassung sind vor allem diese Fähigkeiten nötig:

| Fähigkeit | Konkreter Inhalt |
| --- | --- |
| Allgemeine visuelle Semantik | Gegenstände, Szene, sichtbare Eigenschaften und bei entsprechenden Aufgaben Schrift/Symbole; relevante feine Bilddetails zugänglich halten. |
| Räumliche Beziehungen und Bindung | Welche Hand gehört zu welcher Person; welche Tasse steht auf welchem Tisch; Nähe, Kontakt und Halten unterscheiden. Vollständiges metrisches 3D ist kein Muss für jede Bildfrage. |
| Zeitlicher Zusammenhang | Identität, Bewegung, Kameraänderung, Verdeckung und Zustandswechsel verknüpfen. Dieselbe Endansicht kann aus unterschiedlichen Vorgeschichten entstehen. |
| Ereignisse und Rollen | Wer bewegt welches Objekt, in welcher Reihenfolge und mit welchem beobachteten Ergebnis? Bewegung oder Pose allein legt Absicht und Verursachung nicht eindeutig fest. |
| Evidenzgebundenes Gedächtnis und Abruf | Der aktuelle Weltzustand liest passende zeitgestempelte Beobachtungen/Relationen; zuletzt gesehen, jetzt sichtbar und nur vermutet bleiben unterscheidbar. |
| Vorhersage und Aufgabenbezug | Aus einem bisherigen Verlauf mögliche nächste Zustände ableiten, Fragen auf sichtbare/historische Entitäten beziehen und bei Bedarf gezielt neue Details lesen. Vorhersagen separat gegen einfache Referenzen testen. |

Diese Punkte beschreiben Lernaufgaben und Repräsentationsanforderungen, keine
Verpflichtung zu sechs getrennten großen Netzen. Gemeinsame Multiskalenmerkmale,
der vorhandene Kern/Thinker und Entitätszustände sind passende Anschlusspunkte.
Kleine trainierte Leser können auf dieselbe Quelle zugreifen. Quellauflösung,
Zeitverfügbarkeit, Modell-/Evidenzversion und begrenzter Arbeitssatz bleiben
explizit. Die Engine kann Aufrufe puffern; die Auswahl relevanter Beobachtungen,
Relationen, Aktualisierungen und Antworten muss selbst gelernt/geprüft werden.
Bei Lernmethoden sind vortrainierte Merkmale oder separat erzeugte Teacherziele
optionale Startpunkte; keine Ablösung der akzeptierten gemeinsamen Lernrichtung.

Prüfbares Beispiel: Person A hebt eine Tasse vom Tisch und stellt sie später
hinter eine Box. Aufgaben: Akteur/Objekt binden, Anheben von bloßem Berühren
unterscheiden, letzte beobachtete Position abrufen, bei Verdeckung den aktuellen
Zustand nicht als neu beobachtet ausgeben. Andere Personen, Tassen, Hintergründe
und Kamerabewegungen dienen als getrennte Quellen. Antwort muss auf relevante
Änderungen reagieren, nicht auf bloße Oberflächenwechsel. Physikalische/kausale
Behauptungen benötigen eigene Eingriffs-/Aktionsdaten; reine Reihenfolge ist
kein kausaler Beweis. Audio kann zusätzliche Evidenz liefern, ist aber keine
Pflichtvoraussetzung für visuelle Aufgaben.

Wichtigstes noch fehlendes Element ist die gelernte, nachweislich verwendete
Verbindung der Bausteine. Die [reguläre Suite](understanding-suite-plan.md)
zeigte für die zwei untersuchten Baselines 0/25 bestandene Schnellprüfungen;
eine [gezielte Fortsetzung](grounded-readout-plan.md) bestand 1/25, eng begrenzt
auf kontrollierte Videoreihenfolge, mit Seed-/Erhaltungsgrenzen. Das sind datierte
Versuchsstände, keine neue Evaluation des aktuellen Gesamtsystems und kein
Beweis, dass seine Encoder keinerlei nützliche Information enthalten. Vorhandene
Speicher-/Restart-Tests belegen Mechanik, kein natürliches Personen-/Objektverstehen.

Kleinster vorgeschlagener nächster Fähigkeitsumfang: wenige reale Objektarten,
eine Person-Objekt-Interaktion, zeitlicher Zustand und tatsächlich genutzter
Gedächtnisabruf. Quellen getrennt halten; Paare mit gleichen Objekten, aber
verschiedenen Rollen/Verläufen, sowie erforderliche-Quelle-entfernt- und
Einzelbildkontrollen verwenden. [TempCompass](https://github.com/llyx97/TempCompass)
verwendet widersprechende Videos gegen Einzelbild-/Sprachprior-Abkürzungen;
[Vinoground](https://vinoground.github.io/) ist ein Referenzbenchmark für zeitliche
Komposition. Beides methodische Referenz, kein Download oder ausgewählter Testlauf.
Endaufgaben, strukturierte Zustände und diagnostische Probes getrennt messen;
vor Ausführung Qualitäts-/Erhaltungsgates, Daten- und 6-GiB-Zielbudget samt
End-to-End-Latenz festlegen. Keine neue Implementierung, Trainingsfreigabe,
Modellwahl oder Validierung durch diese Erklärung.

### Ablauf und Segmentierung pro Frame

Direkte Folgefrage: Muss dafür jeder Pixel in jedem Frame segmentiert werden?
Nein: visuelle Merkmalsberechnung und explizite pixelweise Objekt-/Klassenmasken
sind unterschiedliche Rechenschritte. Eine geteilte Bildrepräsentation kann
Detektion, Lage/Attribute, Beziehungen und zeitliche Leser speisen, ohne für alle
Pixel eine semantische Maske auszugeben. Genauere Masken sind ein optionaler
beziehungsweise aufgabenabhängiger Leser, etwa für Objektgrenzen, Flächen oder
Verdeckung; sie allein beweisen keine Interaktion oder physikalischen Kontakt.

Vorgeschlagener kausaler Ablauf innerhalb der Engine:

1. Aktuelles Bild als neue Quelle mit Zeit/Geometrie erfassen; gemeinsame
   Multiskalenmerkmale für die aktiven Leser bereitstellen. Eine günstige Übersicht
   hält auch bisher unbemerkte Bereiche zugänglich.
2. Bestehende Objekte mit aktuellen Merkmalen/Bewegung zuordnen und Zustände
   aktualisieren. Regelmäßige bzw. ausgelöste vollständige Suche nach neuen
   Objekten bleibt nötig; bloß alte Tracks fortzuschreiben kann Neues übersehen.
3. Für die aktuelle Aufgabe, neue Objekte oder schwache Zuordnung zusätzliche
   Detailausschnitte, Masken, Hände/Gesichter oder Geometrie anfordern. Szenenwechsel,
   Verdeckung und widersprechende Evidenz lösen erneute Prüfung aus.
4. Objekt-/Personenbezüge mit dem Verlauf kombinieren: Lageänderung, Rollen,
   Relationswechsel und Ereignis schätzen. Beispiel Tasse: Annäherung der Hand,
   mögliche Aufnahme, gemeinsame Bewegung und neue Ablageposition unterscheiden.
5. Beobachtete Resultate mit Zeit und Quelle speichern; fortgeschriebene/verdeckte
   Zustände als Schätzung führen. Spätere Fragen lesen die relevante Evidenz.

Bereits berechnete Merkmale desselben Bilds sind zwischen kompatiblen Lesern
teilbar. Merkmale eines alten Frames sind bei Bildänderung nicht unverändert als
frische Evidenz verwendbar; Tracking/Maskenpropagation ist eine eigene Inferenz.
[SAM 2](https://github.com/facebookresearch/sam2) ist eine Primärreferenz für
promptbare Videosegmentierung mit Streaming-Gedächtnis und fortgeführten Masken.
Das ist weiter Rechenarbeit, kein kostenloses Wiederverwenden und keine Auswahl
für PATH-WM oder Echtzeitzusage auf RTX3050.

Quellinformation und teure Auswertung bleiben getrennt: Eine kleinere Übersicht
spart Arbeit, kann aber kleine Objekte übersehen. Feinere verfügbare Quellmerkmale
bzw. Quellen müssen für gezielte Detailfragen erhalten/erneut zugänglich sein;
nie erfasste oder verworfene Details kann späteres Zoomen nicht zurückholen.
Die akzeptierten Pre-C-Exporte werden nicht durch eine pauschale verkleinerte
Quelle ersetzt. Sampling, explizite Verluste und ausgelagerte Quelle haben eigene
Budgets. Schnelle Lippen-/Handereignisse verlangen höhere zeitliche Auflösung
als langsam wechselnde Körperform. Kein fixes Segmentierungsintervall festgelegt.

Kleinster Zusatzvergleich: dieselben Aufnahmen und trainierten Leser, Vollauswertung
als Referenz versus bedingte Detailauswertung; neue/kleine Objekte, kurze Ereignisse,
Drift und tatsächliche Relationsantworten sowie Peak-Speicher/p95-Latenz messen.
Eine niedrigere Aufruffrequenz ist nur bei gemessenem Qualitäts-/Latenzkompromiss
sinnvoll. Kein neuer Algorithmus oder Profilinglauf in dieser Diskussion.
