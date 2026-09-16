# Weitere Referenzen: gezielt verbessern, nur nach Nachweis ersetzen

Stand: 16. September 2026. Ergänzung zum
[multimodalen Referenzentwurf](multimodal-reference-design.md).
Auftrag: weitere Quellen unabhängig vom Videokanal suchen, mit Claude prüfen und
nur klar bessere Alternativen übernehmen. **Recherche abgeschlossen; kein neuer
Modellvergleich, kein geänderter Modellstandard.**

## Ergebnis für unsere Architektur

Die zusätzliche Literatur rechtfertigt derzeit keinen Komplettaustausch. Sie
liefert konkretere Kandidaten für drei vorhandene Grenzen: Encoder → Kern,
Zustand → zukünftiger Zustand und Zustand → Ausgabe. Priorität haben kleine
vergleichbare Änderungen an diesen Grenzen, nicht weitere unabhängige Module.

Wir behalten den multimodalen latenten Denkraum, den gemeinsamen Bild-/Videocodec
als optionalen Pfad, räumliche Detailrepräsentationen, austauschbare Adapter und
den versionierten Entity-/Ereignisspeicher. Das ist eine Entscheidung für Anschluss-
fähigkeit und überprüfbare Änderungen, **kein Nachweis ihrer Überlegenheit**.

Der Forschungsplan erhält zwei engere Kandidaten:

1. **Mehrere vorhandene Verarbeitungsebenen auslesen**, statt nur die letzte.
   Zuerst kontrollieren, ob das gegenüber dem bestehenden Feature-Pyramid-/Query-
   Pfad überhaupt etwas ergänzt. Keine zweite Hierarchie daneben bauen.
2. **Aufgabenrelevante zukünftige Merkmale vorhersagen**, unabhängig davon, ob ein
   Decoder bereits schöne Bilder ausgeben kann. Den bestehenden Predictor gezielt
   trainieren/prüfen; keinen zweiten Weltmodell-Kern vorsorglich einführen.

Das Größenverhältnis „kleine Modalitätszweige, größerer gemeinsamer Kern“ bleibt
eine Hypothese. Ein günstiger Rekonstruktionscodec kann mit einem teuren Generator
oder Semantikmodell verbunden sein. Jede Größenangabe braucht diese Systemgrenze.

## Neue Quellen und ihre tatsächliche Aussage

Primärquellen: Originalarbeiten einschließlich Methoden/Anhängen, Autoren-Code und
Autoren-Projektseiten. Die folgenden Erfolge stammen aus den jeweiligen Arbeiten,
nicht aus PATH-WM. Die letzte Spalte ist unsere Übertragungshypothese.

| Bereich / Quelle | Mechanismus und belastbarer Bezug | Grenze und Entscheidung für uns |
| --- | --- | --- |
| Bildmerkmale: [RAEv2](https://arxiv.org/abs/2605.18324), [Code](https://github.com/nanovisionx/RAEv2) | Die letzten k Schichten eines vortrainierten Bildencoders werden addiert. Ein trainierter Decoder nutzt die ergänzenden Merkmale; die Arbeit verbessert Rekonstruktion und Generierung. | Schichten sind nicht automatisch verschiedene räumliche Skalen. Unser Multiscale-Abgriff ist eine eigene Übertragungshypothese. **Erster Vergleichskandidat**, vorhandener Encoder bleibt. Das vollständige Generierungsexperiment nutzt 4 × 8 H100; daraus folgt kein 8-GB-Fit. |
| Trainingssignal: [REPA](https://arxiv.org/abs/2410.06940) | Eine Projektion generativer Zwischenmerkmale wird auf Merkmale sauberer Bilder eines vortrainierten Lehrers ausgerichtet. Das unterstützt die Repräsentationsbildung beim Diffusionstraining. | **Separater späterer Loss-Vergleich**, nicht die Erklärung für Schichtaggregation. Lehrermodell und Vortraining kosten Ressourcen; die publizierte Beschleunigung gilt nicht automatisch für unseren kleinen rekurrenten Kern. |
| Kleiner semantischer Bildzweig: [MobileCLIP2](https://arxiv.org/abs/2508.20691), [Autoren-Tabelle](https://github.com/apple-aiml-research/ml-mobileclip) | Destillation und angereicherte Bild-Text-Daten ermöglichen kleine Bild-/Textencoder. S0: 11,4M Bild- plus 63,4M Textparameter. | **Vortrainierte Referenz oder optionaler Lehrer**, kein sofortiger Codec-Ersatz. Die Tabelle nennt 13 Milliarden gesehene Trainingsbeispiele, nicht ebenso viele unterschiedliche Bilder. Gute globale Bild-Text-Zuordnung beweist keine präzise Richtung, Objektidentität oder Rekonstruktion. iPhone-Latenz ist keine RTX3050-Messung. |
| Kleiner trainierbarer Weltkern: [LeWorldModel](https://arxiv.org/html/2603.19312v1), [Autoren-Code](https://github.com/lucas-maes/le-wm) | Encoder und aktionskonditionierter Prädiktor lernen nächste Embeddings. SIGReg wirkt dem Zusammenfallen aller Repräsentationen entgegen. Die Autoren berichten ungefähr 15M Parameter und begrenzte Kontroll-/Physiktests. | **Priorisierter Trainingsvergleich**, kein Ersatz des Gesamtmodells. Anhang: L40S, Batch128, vier 224²-Frames, gruppierte Aktionen zwischen Frames. Der kompakte CLS-Zustand ist kein hochauflösender Bildcodec. 8-GB-Training muss separat profiliert werden. |
| Gegenprüfung: [unabhängige LeWM-Reproduktion](https://arxiv.org/abs/2608.10145) | Reproduziert TwoRoom und prüft Normalisierung, Aktionszusammenfassung und Zielkonstruktion. Dieselben veröffentlichten Gewichte liefern im Bericht je nach Zielkonstruktion 84% beziehungsweise 8%. | **Protokollkontrolle übernehmen.** Die Arbeit selbst nennt nur einen Seed und 50 Planungsepisoden als Grenzen. Keine robuste Rangfolge ihrer neuen Gewichte daraus ableiten. Eine kleine Ein-Schritt-Loss reicht nicht als Planungserfolg. |
| Alternative Regularisierung: [Sub-JEPA](https://arxiv.org/abs/2605.09241) | Beschränkt Gaussian-Regularisierung auf zufällige Unterräume, um den Zwang im vollständigen Merkmalsraum zu lockern; berichtet Vorteile auf vier Kontrollumgebungen. | **Zurückstellen**, bis ein entsprechender Regularisierungsengpass gemessen ist. Nicht gleichzeitig Zielverlust, Encoder und Regularisierung austauschen. Nicht pauschal auf unsere kategorischen Zustände übertragen. |
| Vorhersage ohne Pixeldecoder: [DINO-WM](https://arxiv.org/html/2411.04983v1) | Sagt räumliche DINOv2-Merkmale aus Historie und Aktionen vorher; optimiert Aktionsfolgen zu Zielmerkmalen. Bilddecoder dient optional der Visualisierung. | **Vergleichsziel für Prediction.** Starker vortrainierter Encoder und aktionsbeschriftete Trajektorien sind Voraussetzungen. Das Verfahren ersetzt weder allgemeine Ausgabeerzeugung noch multimodales Gesprächsverstehen. |
| Begrenzte Planung: [TD-MPC2](https://arxiv.org/abs/2310.16828), [Code](https://github.com/nicklashansen/tdmpc2) | Latente Dynamik, Reward-/Wertlernen und wiederholte lokale Aktionsoptimierung; offizieller Ein-Aufgaben-Standard etwa 5M Parameter. | **Spätere Kontrollreferenz.** Reward, Aktionsraum und Umgebung müssen definiert sein. Kein Grund, den allgemeinen Thinker durch einen kontinuierlichen Motorikplaner zu ersetzen. Verfügbare Modellgrößen beweisen keinen vollständigen Trainingsfit. |
| Text aus gemeinsamem Latentraum: [SONAR](https://arxiv.org/abs/2308.11466), [Code](https://github.com/facebookresearch/SONAR) | Sprach- und Texteingaben werden in einen gemeinsamen Satzraum gebracht; ein Textdecoder kann aus dem Embedding wieder Text erzeugen. Die Schnittstelle nutzt 1024 Dimensionen. | **Positive Referenz für die gewünschte Arbeitsteilung**, nicht für einen winzigen universellen Gesprächsdecoder. Exakte Namen, Zahlen, Negation und Rollen getrennt prüfen. Multimodales Denken muss deshalb nicht zu einer Folge von Satzembeddings werden. |
| Latente Schleifen: [Universal Transformers](https://arxiv.org/abs/1807.03819) | Wiederholte Anwendung geteilter Verarbeitung kombiniert Attention mit rekurrenter Verfeinerung; untersucht algorithmische und sprachliche Aufgaben. | **Bestehende Schleifen beibehalten und gezielt testen.** Die Arbeit begründet keine allgemeine multimodale Denkfähigkeit. Zusätzliche adaptive Stopplogik nicht ohne Vorteil übernehmen; 0/1/2/4-Schritt- und Compute-Kontrollen bleiben erforderlich. |
| Effizienter Sequenzzustand: [Gated DeltaNet](https://arxiv.org/abs/2412.06464), [Code](https://github.com/NVlabs/GatedDeltaNet) | Kombiniert Vergessen durch Gates mit gezielten Delta-Updates eines neuronalen Speichers; auch hybride Varianten mit lokaler Attention werden untersucht. | **Erst bei gemessenem Sequenzkostenproblem vergleichen.** Sprach-/Retrieval-Benchmarks begründen keinen Austausch unseres Thinkers oder Entity-Stores. Kernelaufwand auf kurzer Sequenz und RTX3050 separat messen; ein interner Zustand ersetzt keine revidierbaren Belege. |
| Audio: [SemantiCodec](https://arxiv.org/html/2405.00233v2) | Vortrainierte AudioMAE-Semantik plus akustischer Rest; ein Diffusionsdecoder rekonstruiert Sprache, Musik und Geräusche. | **Idee getrennter Semantik/Details prüfen, Pipeline nicht übernehmen.** Publizierter Pfad: 10,24-Sekunden-Blöcke und bidirektionale Verarbeitung. Das passt ohne Umbau nicht zu kurzer kausaler Sprachinteraktion. Wenige Tokens bedeuten nicht geringe Decoderkosten. |
| Audio-Kontrolle: [DAC / Improved RVQGAN](https://arxiv.org/abs/2306.06546), [Code](https://github.com/descriptinc/descript-audio-codec) | Faltungsbasierter Audiocodec mit residualer Vektorquantisierung und gelerntem Decoder; Gewichte und Training für allgemeines Audio verfügbar. | **Offline-Rekonstruktionsreferenz**, kein semantischer Audioencoder oder Gesprächssystem. Kausalität/Lookahead sind für einen später gewählten Checkpoint gesondert zu prüfen; nicht als Streamingnachweis verkaufen. |
| Wahrnehmung → Entities: [SAVi++](https://slot-attention-video.github.io/savi%2B%2B/) | Zeitlich fortgeführte Objektslots lernen über Tiefensignale, optional mit Boxen im ersten Frame. Reale Fahrvideos werden in Objektmerkmale zerlegt. | **Kandidatenquelle für Binding**, kein Ersatz für stabile IDs. Laut Autoren kann ein Slot nach dem Verlassen eines Objekts ein anderes übernehmen. Ohne Initialisierung ist Tracking schwächer. Tiefensignale sind zusätzliche Supervision, nicht kostenloses RGB-Verstehen. |
| Graphabruf: [HippoRAG 2](https://arxiv.org/html/2502.14802v2), [Code](https://github.com/OSU-NLP-Group/HippoRAG) | Verbindet Passagen/Entities und Abfragezuordnung mit Personalized PageRank; ein LLM filtert relevante Verknüpfungen. | **Späterer Retrievalvergleich**, vorhandener Store bleibt. Hauptauswertung nutzt Llama3.3-70B und NV-Embed-v2. Graphtraversal allein erbt diese Leistung nicht. Belege, Gültigkeitszeiten und Korrekturen bleiben erforderlich. |

Die Schlüsselfrage lautet nicht „Welches Paper gewinnt?“, sondern „Welche kleine
Änderung beseitigt unseren gemessenen Engpass unter unseren Ressourcen?“

## Konkrete Vergleiche in dieser Reihenfolge

### 1. Gleiche Merkmale, besserer Zugang zum Kern

Zuerst das vorhandene Feature→Core-Diagnoserezept abschließen. Ein gutes direktes
Auslesen bei schlechtem Core-Auslesen spricht für ein Anschlussproblem; die
Umkehrung beweist allein noch keinen unumkehrbaren Informationsverlust.

Danach ausschließlich den Abgriff variieren: bisheriger letzter Abgriff gegen
budgetierte Kombination vorhandener Zwischenebenen. Gleicher Encodercheckpoint,
identische Quellgruppen, gleiches Tokenbudget und gleiche nachfolgende Leser.
Zusätzliche Projektionen, Feature-Caches und Latenz mitzählen. Verschiedene H/W/C
explizit ausrichten; nicht blind unterschiedlich geformte Tensoren addieren.
Ein vorhandener Multiscale-Pfad ist eine eigene Baseline, kein neu erfundenes Modul.

Messen: Farbe, relative Position, Bewegung, Identität und Relationen direkt und
nach Fusion/Belief/Workspace; Codec-Rekonstruktion zusätzlich als Regression.
Ein Adapter, der nur Bild-Text-Ähnlichkeit verbessert und Richtung verliert, wird
nicht zum allgemeinen Ersatz. RAEv2-Daten sind Motivation, kein lokaler Nachweis.

### 2. Zukunft im latenten Raum gezielt lernen

Bei eingefrorenem Encoder: kleiner Predictor auf dessen zukünftige Merkmale;
keine neue Encoder-Regularisierung nötig, um einen eingefrorenen Zielencoder vor
Collapse zu schützen. Copy-last-state, konstante Ausgabe, vertauschte Historie
und aktuelle Beobachtung ohne Historie sind Kontrollen. Fehler auch nach mehreren
Schritten und in unabhängig gelesenen Zustandsgrößen beurteilen; der rohe Abstand
zweier unterschiedlich skalierter Latenträume ist keine faire Rangfolge.

Separat: ein kleiner von Grund auf trainierter Encoder/Predictor nach LeWM-Prinzip
als Vergleich für Daten- und Parametereffizienz. Anti-Collapse-Statistik prüfen,
nicht nur Lossabfall. Gaussian-Verteilungsregularisierung über Beispiele ist
**nicht dieselbe Operation wie der KL-Term eines stochastischen VAEs**. Sie wird
nicht ungeprüft auf alle Belief-Codes oder gespeicherten Entities angewendet.

Für Aktionswirkungen brauchen wir protokollierte Aktionen und nachprüfbare
Zustandsübergänge. Gewöhnliche Videoclips erlauben zunächst beobachtende Prediction;
aus ihnen allein folgt keine interventionelle Kontrolle. Erst im kleinen Simulator
mit Aktionslogs Planer gegen echte Zielerreichung testen, dann Transfer prüfen.

Kleinere Batches ändern die Verteilungsstatistik. Gradient Accumulation reproduziert
einen über den gesamten Batch berechneten SIGReg-Loss nicht automatisch. Datennormalisierung,
Frame-Abstände, Aktionsaggregation, Horizont und Zieldefinition ins Runmanifest.

### 3. Ausgaben, Erinnerung und Interaktion am selben Zustand prüfen

Text zuerst auf überprüfbare Inhalte konditionieren: unbekannte Namen/Zahlen,
Negation, Rollentausch, korrigierte Fakten, mehrere Gesprächsrunden. Satzähnlichkeit
allein genügt nicht. Direkter Fakt-/Oracle-Kontext gegen denselben Inhalt im
gemeinsamen Latentzustand trennt Sprachdecoder- von Anbindungsproblemen. Ein
eingefrorener vortrainierter Decoder kann weiterhin selbst schlussfolgern.

Audio separat: Akustikrekonstruktion, Ereignis-/Spracherkennung, zeitliche Zuordnung
und zustandsgetreue Ausgabe. Für interaktiven Betrieb erst zulässigen Lookahead,
Chunkdauer und Zeit bis zur ersten Ausgabe festlegen. Den bereits recherchierten
kausalen Mimi-Ansatz dafür heranziehen; DAC bleibt Offline-Kontrolle, SemantiCodec
vorerst Architekturhinweis. Nicht nebenbei drei große Audiopipelines integrieren.

Bei Memory zuerst Oracle-Kandidaten/Relationen, dann gelernte. Gleicher Abrufumfang
für flaches Retrieval und relationale Erweiterung; Faktenkorrektur, Fehlzuordnung,
Verdeckung/Wiederkehr und widersprüchliche Belege auswerten. Lokale Slots nur über
den Binder an persistente Entity-IDs anschließen. Der Graph bleibt multimodal.

## Wann ist eine Alternative klar besser?

**Paperbefund → lokaler Vergleichskandidat → validierter Ersatz** sind drei
verschiedene Status. Alle neuen Alternativen stehen höchstens beim zweiten.
Vor einem Austausch müssen diese Bedingungen erfüllt sein:

| Bedingung | Vor dem Lauf festzulegen / danach zu belegen |
| --- | --- |
| Relevante Verbesserung | Ein primärer Aufgabenerfolg mit kleinster relevanter Differenz; alternativ weniger Ressourcen bei vorab definierter Nichtunterlegenheit. Pixel-MSE ersetzt keine Identitäts-/Gesprächsprüfung. |
| Vergleichbarkeit | Gleiche Daten/Splits und klar getrennte Läufe für gleiche Updates versus gleiches Zeit-/Compute-Budget; identische Initialisierung, soweit möglich. Fremdes Vortraining und Teacher-Caches als eigene Kosten ausweisen. |
| Generalisierung | Frische Quellgruppen, zurückgehaltene Kombinationen und mindestens drei vorab festgelegte Seeds. Frames eines Clips nicht als unabhängige Stichproben zählen; Unsicherheit auf Quellen-/Seedebene berichten. Drei Seeds sind kein automatischer Signifikanznachweis. |
| Kein verdeckter Rückschritt | Vorab gewählte Pflichtmetriken und Toleranzen für Fakten, Farbe, Geometrie, Zeit, Kausalität, Wiederaufnahme, alte Modalitäten und Memory-Kompatibilität. Kein Mittelwert darf einen harten Fehler verdecken. |
| Hardware | Vollständiges Training inkl. Optimizerstep/Backward und vollständiger Inferenzpfad auf der RTX3050; bisher vorgeschlagenes 6-GiB-Prozessbudget, Laufzeit und Streamingzustand messen. Gewichtsspeicher allein genügt nicht. |
| Reproduzierbare Entscheidung | Festes Lauf-/Tuningbudget, keine nachträgliche Seedwahl oder Endpunktänderung. Entwicklungsdaten wählen den Kandidaten; gesperrter Abschlusstest wird nicht zur Auswahl weiterer Varianten verwendet. |

Ein bestätigender Lauf erhält **einen** primären Endpunkt und feste Regressionen.
Bei mehreren bestätigenden Kandidaten/Behauptungen wird die Testfamilie samt
Verfahren zur Kontrolle mehrfacher Tests vorher festgelegt. Kein Einzeltest pro
Frame oder nachträglich gewähltem Seed. Stopps: ungültige Daten/Gradienten,
Ressourcenüberschreitung oder ausgeschöpftes Budget; ein abgebrochener Qualitätslauf
ist kein Pass. Bei unklarer Differenz bleibt der bisherige Standard.

Die konkreten Qualitätsdifferenzen, Nichtunterlegenheitsgrenzen, Datenmengen und
Laufbudgets sind **noch offen**, weil diese Recherche keinen neuen Fit beauftragt
oder präregistriert. Sie werden anhand der tatsächlichen Aufgabe vor deren Lauf
gesetzt. Bekannte Überlappung mit Vortrainingsdaten offenlegen; vollständige globale
Trennung vom unbekannten Lehrerkorpus nicht behaupten. Neue eigene Kontrollaufnahmen
helfen, diesen Einfluss einzugrenzen.

## Claude-Abgleich und Arbeitsstand

Zwei tatsächliche isolierte Claude-Sonnet-Reviews mit ausschließlich öffentlichen,
hypothetischen Briefs. Claude führte selbst keine Websuche aus; die Primärquellen
wurden hier abgerufen, gelesen und seine Vorschläge daran überprüft. Neuere Titel
konnte Claude nicht aus seinem eigenen Wissensstand bestätigen und hat das benannt.

Nach Rückmeldung akzeptierte Claude sieben Präzisierungen: LeWM-Hardware und
Batchstatistik; Grenzen der unabhängigen Reproduktion; Schichten versus Skalen und
Trennung von REPA; Rücknahme des SemantiCodec-Streamingvorschlags; vollständige
MobileCLIP-Kosten; begrenzte statt universell verlustfreie Textprüfung; Slots und
Graphabruf als Hilfen statt Entity-Store-Ersatz. Sein offener Einwand zu vorab
festgelegten Regressionen/Mehrfachtests ist oben als Voraussetzung aufgenommen.
Eine dritte Runde ohne neue Entscheidung war nicht nötig.
Universal Transformers und Gated DeltaNet wurden anschließend ergänzend anhand
öffentlicher Quellen eingeordnet; dazu wurde kein zusätzlicher Claude-Aufruf gemacht.

Briefs, Antworten, Hashes und Quellenabrufe:
`runs/reviews/architecture_extension_v1/`. Es wurden keine privaten Codes,
Datensätze oder Messwerte an Claude gesendet. Keine Modellgewichte heruntergeladen,
kein Training gestartet, keine Implementierung ersetzt oder Validierungsfarbe
hochgestuft. Nächster Schritt bleibt der vorhandene Feature→Core-Vergleich;
Schichtaggregation ist sein erster begrenzter zusätzlicher Kandidat.
