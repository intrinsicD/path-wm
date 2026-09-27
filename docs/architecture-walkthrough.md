# Architektur von oben nach unten

18. September 2026. Alex möchte zuerst die funktionalen Module und ihre
Zusammenarbeit durchgehen. Danach sollen die bereits besprochenen DeepSeek- und
Jev/Calibrierungsansätze konkreten Stellen zugeordnet werden. Diese Reihenfolge
ist vereinbart; die folgende Gruppierung ist eine Erklärung des bestehenden
Entwurfs, keine neu verabschiedete Architektur.

## Oberste Ebene

| Bereich | Aufgabe | Nächste Unterteilung |
| --- | --- | --- |
| Eingang und Wahrnehmung | Verfügbare Beobachtungen in nutzbare Merkmale überführen | Eingangsadapter, Encoder für Bild/Video/Audio/Text, Merkmale auf mehreren Skalen |
| Gemeinsamer Kern | Den aktuellen Zustand schätzen, Veränderungen vorhersagen und Aufgaben bearbeiten | Beobachtungskorrektur, Dynamikmodell, Thinker und Arbeitszustand |
| Ausgabe und Handeln | Aus internem Zustand eine Ausgabe oder einen Handlungsvorschlag erzeugen | Modalitätsspezifisches Auslesen/Decoder, Aktionsausgabe |

Zwei querliegende Bereiche gehören dazu: **Kontext und externes Gedächtnis**
stellen Informationen bereit; **Aufgabensteuerung und Planung** steuern anhand
von Anfrage/Ziel und Budget, welche Operationen ausgeführt werden. Das ist kein
einmaliger linearer Durchlauf. Eine neue reale Beobachtung kann den Kreislauf
fortsetzen; eine erzeugte oder imaginierte Ausgabe wird dadurch kein neuer Beleg.

Der Dynamikbaustein ist für den nächsten Zustand aus bisherigem Zustand, Zeit und
gegebenenfalls Aktion zuständig. Physikalische Bewegung ist ein Anwendungsfall.
Welche sequenziellen Bild-, Audio- und Textziele welchen Teil trainieren, gehört
zum späteren Trainingsdurchgang; Decoder-Sequenzmodellierung und Weltzustands-
vorhersage dürfen nicht allein wegen des gemeinsamen Wortes „Vorhersage“
gleichgesetzt werden.

## Begriffe für internen und externen Speicher

Alex möchte für den **internen** Speicher die Begriffe **Local Context** und
**Global Context** verwenden. Der externe World State bleibt davon getrennt.
Die Bezeichnung wird im Gespräch und in der Übersicht übernommen. Noch offen
ist, welche vorhandenen Speichergruppen, Zeiträume und Lesezugriffe darunter
fallen. Insbesondere werden `recent`, `staging`, `compressed`, `protected` und
`consolidated` nicht ohne weiteren Durchgang auf zwei Kategorien festgelegt.

Context bezeichnet gespeicherte beziehungsweise bereitgestellte Information.
Schreiber, Kompressoren und Leser sind die darauf arbeitenden Module. Eine
Umbenennung legt weder ein neues Attention-Verfahren noch ein Fenster, einen
Speicherumfang oder Gewichtsänderungen fest. Die bestehenden Code-/Checkpoint-
Namen bleiben vorerst erhalten. Die optionalen World-State-Komponenten sind
implementiert, aber nicht automatisch Teil jeder Modellkonfiguration.

## Kontext nutzen und steuern: aktueller Diskussionsvorschlag

Alex schlägt vor, dass das vorhandene Denkmodul ausgewählte Konzepte, Instanzen
und Komponenten aus dem Knowledge Graph in den internen Kontext holt und dort
auch Zwischenergebnisse ablegt. Er fragt anschließend ausdrücklich, wer Abruf
und Übergänge zwischen Local und Global Context entscheidet. Seine Bestätigung
„genau das meine ich“ klärt zunächst die Identität des Denkmoduls; sie ist keine
pauschale Annahme aller nachfolgend vorgeschlagenen Implementierungsdetails.

**Empfehlung:** Diese Nutzung ist sinnvoll als zu prüfender Entwurf. Local Context
enthält die für den aktuellen Teil der Aufgabe unmittelbar benötigten Details
und Arbeitswerte. Global Context hält ausgewählte, über mehrere Denkschritte
benötigte Informationen der laufenden Aufgabe bereit. Global bedeutet hier weder
unbegrenzten Speicher noch automatisch dauerhaftes Wissen über alle Sitzungen.
Konkrete Lebensdauern, Größen und Reset-Regeln sind noch auszuwählen.

Ein Eintrag kann einen ausgewählten Graph-Bestandteil mit Referenz darstellen,
ein erhaltenes Beobachtungsmerkmal oder eine abgeleitete Hypothese. Konzept und
Instanz bleiben unterscheidbar; der Abruf einer Instanz lädt nicht automatisch
alle Komponenten, Nachbarn und Konzepte. Identität, Komponentenauswahl und
Relevanz müssen zusammen mit dem aktuellen Ziel berücksichtigt werden.

### Wer entscheidet?

**Angenommene Richtung:** Alex bestätigt anschließend mit „Ja, gute Idee“ die
gezielte Erweiterung der vorhandenen TaskPolicy um diese Kontextentscheidungen.
Die Bestätigung bezieht sich auf diese Zuständigkeit, nicht auf bereits
implementiertes Verhalten, sämtliche Ausführungsdetails oder eine Rechenfreigabe.

| Teil | Zuständigkeit |
| --- | --- |
| Denkmodul | Verarbeitet Aufgabe und verfügbaren Kontext; sein Zustand liefert die Grundlage für weitere Kontextentscheidungen. |
| Kleiner trainierbarer Steuerkopf an der Aufgabensteuerung | Liest Denkzustand, Ziel und Kontextmetadaten; schlägt Operation, Suchanfrage, Kandidatenauswahl, Zielbereich und Priorität vor. Kein weiterer allgemeiner Denkapparat. |
| Retriever und Kontextencoder | Finden erlaubte Kandidaten im Graphen und machen ausgewählte Komponenten/Relationen für den Kern lesbar. Auffinden eines Kandidaten ist noch keine Entscheidung, ihn langfristig zu behalten. |
| Kleine ausführende Speicherverwaltung | Wendet Vorschläge innerhalb fester Kapazitäts-, Quellen-, Zeit- und Versionsregeln an; meldet Ablehnung oder ausgelassene Inhalte. Sie garantiert diese Verträge, nicht die inhaltliche Richtigkeit oder Relevanz. |

Der Steuerkopf wäre eine gezielte Erweiterung der bestehenden TaskPolicy bzw.
ein kleiner zugehöriger Ausgabekopf. Die Zuständigkeit ist angenommen; die
konkrete Kopfform und Aktionsschnittstelle sind **Implementierungsvorschläge**.
Aktuelle grobe Operationswahl und diese detaillierten Entscheidungen sind
unterschiedlich. Zustandsübergänge bei neuen realen Beobachtungen sowie zwingende
Invalidierung veralteter Referenzen können weiterhin deterministisch erfolgen.
Das Modell muss solche Gültigkeitsregeln nicht erst lernen.

| Vorgeschlagene Aktion | Bedeutung |
| --- | --- |
| Graph → Local oder Global | Suchanfrage stellen, erlaubte Kandidaten auswählen und für die gewünschte Lebensdauer verfügbar machen. |
| Global → Local | Ausgewählte Information für den aktuellen Schritt aktivieren oder auslesen; normalerweise weder aus Global löschen noch eine zweite dauerhafte Kopie anlegen. |
| Local → Global | Information oder ein gekennzeichnetes Zwischenergebnis für weitere Schritte behalten; keine automatische Aufwertung zur Beobachtung oder bestätigten Wahrheit. |
| Freigeben / ersetzen | Eintrag aus einem Kontextbereich nehmen; das löscht nicht automatisch den Graph-Datensatz. Auslassungen und Verdrängungen bleiben begrenzt protokolliert. |
| Kontext → dauerhafter Graph | Eigener versionierter Schreibvorschlag mit Herkunft und Rolle; eine Inferenz darf als Inferenz gespeichert werden. Dies ist kein Nebeneffekt eines Abrufs. |

Als erste kleine Umsetzung bietet sich **ein begrenzter Bestand von Referenzen
mit Local-/Global-Zuordnung und Prioritäten** an. Zwei logische Sichten benötigen
keine doppelte Datenbank. Ein Quellobjekt kann in beiden Sichten erreichbar sein;
mutierbare Arbeitswerte sind getrennte Ableitungen. Bei einer Quellkorrektur
müssen auch davon abhängige Zusammenfassungen und Hypothesen transitiv überprüft,
invalidiert oder neu erzeugt werden. Herkunft darf nicht nur in einem latenten
Vektor vermutet werden: Referenzen und Versionen bleiben explizite Metadaten.
Ein Zusammenfassungsvektor ersetzt verlorene Quelldetails nicht.

### Heutiger Code und noch fehlender Teil

`WorldSession.think(query, ...)` erhält die Anfrage heute vom Aufrufer, ruft den
Retriever auf, codiert Komponenten/Relationen und übergibt die Tokens an den
Thinker. Dessen Arbeits-/Reasoning-Tokens ändern sich, der Graph wird dadurch
nicht geschrieben. Ein optionaler `QueryGenerator` erzeugt gelernte Suchschlüssel;
das allein ist keine gelernte Entscheidung über Zeitpunkt, Zielbereich und
Aufbewahrung. Die bestehende `TaskPolicy` wählt unter anderem `think` und `recall`;
der aktuelle `recall`-Dispatcher liest den Sitzungsspeicher. Eine vollständige
autonome Graph-/Local-/Global-Steuerung existiert damit noch nicht.

Quellen: [WorldSession](../pathwm/world_state/session.py),
[Graph-Module](../pathwm/world_state/modules.py),
[TaskPolicy](../pathwm/models/tasks.py), [Dispatcher/Thinker](../pathwm/models/agent.py).

### Lernen, Grenzen und kleinster Vergleich

Die Speicherwahl muss mittrainiert oder zunächst explizit vorgegeben werden.
Für einen ersten kontrollierten Lernvergleich sind bekannte relevante Quellen,
notwendige Aufbewahrung über Ablenkungen und nachvollziehbare Aufgabenresultate
geeignet. Differenzierbare Leser/Kandidaten-Scores können Aufgabenverluste nutzen;
harte Auswahl und Abrufoperationen erhalten nicht automatisch gewöhnliche
Backpropagation durch alle Entscheidungen. Zunächst passende Auswahlziele bzw.
vorgegebene Abläufe verwenden; RL ist eine spätere Option für begründete sequentielle
Kosten/Nutzen-Ziele, keine Voraussetzung für den ersten funktionierenden Pfad.
Zielherkunft und Datenaufteilung müssen nachvollziehbar sein. Ein Trainingsoracle
darf spätere Ergebnisse seiner Trainingstrajektorie zur Zielbildung verwenden;
verborgene zukünftige Inhalte dürfen dabei nicht zu Entscheidungseingaben werden.
Auswertungsantworten und spätere Testkorrekturen dürfen nicht ins Training gelangen.
Vorgegebene Relevanzlabels können die Auswahlpräferenzen ihrer Konstruktion erben.

Vergleiche zunächst feste mit gelernter Auswahl bei gleicher Kontextschnittstelle,
gleicher Gesamtzahl gespeicherter/gelesener Tokens und gleichem Abruf-/Arbeitsbudget;
Rechen- und Abstimmungsaufwand beider Verfahren ebenfalls deklarieren und abgleichen.
Eine flache priorisierte Sicht bleibt ein Vergleich für den Nutzen der zwei
Geltungsbereiche. Aufgaben sollten Ablenkungen, später erneut benötigte Details
und Quellkorrekturen enthalten. Getrennt prüfen: Aufgabenqualität, falsche/stale
Quellenverwendung, Auswahl-/Verdrängungsfehler, erhaltene Herkunft, Speicher und
Laufzeit. Rollenlabels und Invalidierung allein garantieren weder inhaltlich treue
Zusammenfassungen noch belegtreues Auslesen. Inferenzinhalte dürfen weitere
vorläufige Überlegungen oder Aufbewahrung begründen; Beobachtungsstatus und
Ausführungsbefugnisse entstehen daraus nicht. Noch kein Datensatz, numerischer
Grenzwert oder Rechenbudget ausgewählt.

Dies konkretisiert die stehenden Prinzipien: klare Zustandsbesitzer/Lebensdauern,
Quelle von Ableitung unterscheiden, Vorschläge von Vertragsprüfung trennen,
Information einmal halten und gezielt lesen sowie dieselben Zugriffsbeschränkungen
trainieren und auswerten. Erwarteter Nutzen sind weniger redundante Kopien und
gezielter Zugriff; Kosten sind Auswahlaufwand, Aktualisierung von Abhängigkeiten
und das Risiko, später benötigte Informationen auszublenden. Bessere Qualität
oder Geschwindigkeit ist dadurch noch nicht nachgewiesen.

[MemGPT](https://arxiv.org/abs/2310.08560) ist ein veröffentlichtes Beispiel für
modellgesteuerten Zugriff auf verschiedene Speicherebenen bei Sprachmodellen.
Das ist ein verwandtes Prinzip, kein Nachweis für diesen latenten Graph-Kontext,
die vorgeschlagenen Lebensdauern oder eine optimale Zweiteilung.

Öffentliche, generische Methodik wurde mit tatsächlichem Claude gegengeprüft;
keine privaten Quelldateien oder Messwerte wurden exportiert. Quellkorrekturen
müssen transitive Ableitungen erfassen; Auswahlqualität und Kreditzuweisung
bleiben die hauptsächlichen offenen Lernprobleme. Ein gemeinsamer Referenzbestand
mit Bereichsmarkierungen ist die einfache erste Umsetzungsidee. Die Rücksprache
präzisiert außerdem vergleichbaren Rechen-/Abstimmungsaufwand, die Herkunft von
Auswahlzielen und den Unterschied zwischen Herkunftsmetadaten und semantischer
Belegtreue. Scope gegenüber reiner Priorität bleibt eine empirische Frage. Receipts:
`runs/reviews/context-scopes-20260918-205320/`. Keine Implementierung oder neue
empirische Validierung durch diese Diskussion.

## Completion priorities after the concept and conversation discussion

24 September 2026. Alex asks what to work on next to complete the architecture.
This is a proposed order for closing the design, not authorization of a new
training budget or evidence that the complete agent works.

The conversation clarifies the intended use: reusable knowledge of how a class
varies should combine with details of a new instance and its requested/current
state. Human pose/view synthesis is an example, not the sole application or a
selected first dataset. Evidence should accumulate and revise uncertain estimates;
use the best available estimate when needed without relabelling it as observation.
Better reconstruction remains an explicit objective. Conversation should maintain
continuity in text/audio; ordinary replies need not invoke multi-step planning.

Terminology correction: **Local and Global Context are both internal context** as
defined above. Persistent knowledge belongs to the external World State/store.
Earlier assistant wording in the conversation blurred this boundary. A proposed
conversation episode stores events and versioned state; active contexts load
selected working values with references, not just an unusable graph pointer.
This does not require a third peer context system. Conversation-specific state,
speech timing and learned language grounding are not established by current code.

### Close these interfaces before choosing more mechanisms

1. **Concept, instance and current state.** Define what reusable structure supplies,
   what instance detail must survive and what changes with time/view/pose. Keep
   source evidence and uncertainty attributable to the affected details. These are
   functional roles, not a requirement for three networks or perfectly disentangled
   tensors. Some reusable knowledge can live in weights; acquired concept codes and
   examples can live in memory. Their concrete representation remains open.
   For remembered instances: a slowly changing identity part belongs to the
   instance record; pose, expression, lighting and background are per-observation
   state with their own versioned ownership. A learned (paired views, integrated
   plan §19) versus a partly parametric (FLAME/SMPL-X, human-perception discussion)
   identity split is an open comparison ([ideas](compact-instance-memory-ideas.md), 27 September).
2. **Evidence to memory.** Declare association, update, contradiction, source
   correction and actual state-change behavior. Preserve previous evidence while
   revising estimates and invalidating derived state. Generated completions can be
   useful hypotheses, never independent confirmation. Extend existing session/store
   ownership instead of introducing another authoritative memory. Confidence fields
   and revision mechanics alone do not establish calibrated learned uncertainty.
3. **Memory to active context.** Specify selection, capacity, source/version
   references, eviction, reset and resumption. A conversation is a persistent
   episode with utterance events and revisable derived working state. Exact wording
   and numerical facts remain retrievable. Referenced state must be compatible with
   its model version. Existing TaskPolicy ownership of context decisions stands;
   detailed learned retrieval and retention are still unproven.
4. **Active context to response.** Start with receive → update → select/retrieve →
   optional refinement → respond or wait. Keep world time and refinement iterations
   separate. Clarification, interruption and response completion need contracts;
   bounded multi-step planning remains optional for tasks requiring it. Text and
   audio should use the same conversational state while retaining modality-specific
   generation and streaming buffers. Direct latent speech remains a candidate;
   this discussion does not mandate a text bridge.
5. **Learning and evaluation.** Specify how each operation receives supervision:
   identity/detail retention across views, reconstruction, transformation/next-state
   prediction, evidence revision and conversation continuity. Do not add all losses
   at once. Common tensor width is not evidence of semantic compatibility. Define
   splits, baselines, quality gates, seeds and compute budgets before any run.

### First complete demonstration to specify (proposal)

Continuation update: the [bounded context slice](context-retrieval-plan.md) adds versioned
flat/Local–Global pins and optional TaskPolicy lexical ranking. Two seeds pass held-out
alias pairings, but counted lookup matches them and context reset/swap changes no answers.
Useful learned retention and autonomous graph access remain open. This implementation
does not complete the pending user walkthrough.

Implementation update: the [evidence-loop slice](evidence-loop-plan.md) now passes
two-seed controlled RGB16 reconstruction/revision gates, with session and reactive
episode mechanics tested. Learned dialogue, natural-human views and general concept
induction remain open; this does not complete the entire agent.


Use a controlled visual entity with an observable hidden detail: observe a partial
view, store instance evidence, recall it after interruption, request another state
or view using learned shared structure, then reveal the detail and revise only
the relevant estimate. Hold test instances out of training. A requested
transformation is distinct from predicting which motion will actually occur.
The posterior must not be scored as confidently wrong for initially unknowable
detail; evaluate uncertainty and its revision separately from visible-detail
reconstruction. Compare against retrieval/copy and no-update controls so a demo
cannot pass solely by storing frames or substituting a supplied identity.

Measure identity/detail retention, requested-output quality, evidence-dependent
correction, unrelated-knowledge retention, restart behavior, latency and memory.
No numerical thresholds or budget are selected here. Text conversations about the
same entities can follow, including corrections and references across turns, then
streaming audio; success on the visual test alone establishes neither capability.
This proposal does not erase failed concept transfer or adaptive-retention results.

Prepare source features once, retain accessible detail separately from compact
state, give writes explicit provenance, and measure useful progress per real cost.
Those standing principles motivate reusing the current library and recipes.
The cost of retained detail, retrieval and repeated computation must be included;
latent operation is not intrinsically faster. Consequential method adoption still
requires the existing independent review and predeclared experiment workflow.

## Spätere Zuordnung der Techniken

Der [Aktionsdurchgang](action-semantics-design.md) behandelt inzwischen die
Unterscheidung von Beobachtung, bekanntem Aktionslog, inferierter Aktion,
Instruktion und internem Vorschlag sowie Ausführbarkeit, Planung und Rückmeldung.
Die Kontext-TaskPolicy ist ein Teil dieser Operationssteuerung; neue Schnittstellen
und Fähigkeitsprüfungen bleiben Vorschläge.

Für jeden Kandidaten festhalten: konkreter Baustein und Aufrufpfad, beabsichtigter
Nutzen, Voraussetzungen, veröffentlichte Evidenz, örtlich noch ungeprüfte
Übertragung und kleinster sinnvoller Vergleich. Ausgangspunkte sind die
[DeepSeek-Übertragung](deepseek-v41-transfer-proposal.md) und die
[Jev/Calibrierungsbewertung](decision-design.md#calibration-training-and-the-jevrlcd-comparison).
Nach dem bisherigen Quellenstand ist Jevs konkrete RLCD-Trainingsrezeptur nicht
offengelegt; die veröffentlichten verwandten Verfahren dürfen ihr nicht
zugeschrieben werden. Ergebnisse eines Gesamtsystems isolieren auch nicht
automatisch den Nutzen jeder Einzeltechnik für PATH-WM.

Angewandte Gestaltungsprinzipien: Zustände mit klaren Besitzern/Lebensdauern
darstellen und erhaltene Information, gelesene Information sowie wiederverwendbare
Berechnungen unterscheiden. Das macht die Zuständigkeiten nachvollziehbar, ohne
bereits zusätzliche Speicher oder Cache-Schichten einzuführen. Kleinster nächster
Abgleich: einen realen Beobachtungs- und anschließenden Fragepfad durch die
vorhandenen Aufrufe verfolgen. Qualität, Speicherbedarf und Laufzeit werden erst
bei einer konkret ausgewählten Änderung verglichen.

Noch keine vollständige Nutzer-Durchsprache, Modelländerung, neue Messung oder
Fähigkeitsvalidierung. Die bestehenden begrenzten Mechaniknachweise bleiben in
der [Architektur-Checkliste](architecture-discussion.md) getrennt erfasst.
