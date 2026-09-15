# Literaturrecherche: Richtung im gemeinsamen latenten Zustand

Stand: 15. September 2026. Gezielte Recherche in Originalarbeiten und offiziellen
Proceedings; keine systematische Vollständigkeitsbehauptung. Abstracts und die unten
benannten Methodenstellen wurden geprüft. Neue Modellversuche wurden nicht gestartet.

**Ergebnis:** Es gibt publizierte Lösungsansätze für die beobachteten Problemklassen.
Keine dieser Arbeiten beweist bereits die Ursache oder Reparatur unseres Fehlers.
Am nächsten liegen kategorisches Lernen, getrennte Lernsignale für verschiedene
Faktoren und die Erhaltung mehrerer gelesener Merkmalsvektoren. Unser gemeinsamer
multimodaler Denkraum muss dafür nicht durch textbasiertes Denken ersetzt werden.

## Ausgangspunkt im tatsächlich getesteten Modell

Die [lokalen Ergebnisse](modality-readout-plan.md#direction-follow-up-what-is-actually-failing)
zeigen teilweise auslesbare Richtung vor der Code-Auswahl, schwache Nutzung danach
und Fehler des mittrainierten Lesers schon davor. Die Encoderdiagnose am ursprünglichen
Checkpoint und die spätere Diagnose der reparierten Checkpoints sind getrennte Vergleiche.

Der aktuelle [Trainingspfad](../experiments/modality_readout.py) mittelt drei
Kreuzentropieverluste für Farbe, Position und Richtung. Optional kommt ein Hilfsverlust
vor dem Sampling dazu. In dieser Core-Aufgabe gibt es **keinen KL-Verlust**. Der
[Zustands-Updater](../pathwm/models/belief.py) liest mit Attention, mittelt anschließend
die Query-Vektoren und projiziert auf vier Gruppen mit je acht Kategorien.
`draw()` verwendet harte Stichproben vorwärts und den Gradienten der Wahrscheinlichkeiten
rückwärts. Das ist kein Nearest-Neighbor-VQ-Codebuch. Positions-/Zeitsignale sind in den
[Encoderfeatures](../pathwm/models/multiscale.py) bereits vorhanden.

Daraus folgen drei Grenzen der Übertragung: KL-bedingter Posterior Collapse ist hier
nicht nachgewiesen; VQ-Codebuchverfahren passen nicht unmittelbar; zusätzliche
Positionskodierung ist keine belegte fehlende Grundkomponente.

## Geprüfte Veröffentlichungen und konkrete Relevanz

| Arbeit | Publizierter Befund / Verfahren | Übertragung auf PATH-WM und Grenze |
|---|---|---|
| **Jang, Gu, Poole: Categorical Reparameterization with Gumbel-Softmax, ICLR 2017.** [Original, §2.1–2.2](https://arxiv.org/abs/1611.01144) | Differenzierbare Relaxation kategorischer Stichproben; ebenfalls enthalten: harte Vorwärtsauswahl mit Gumbel-Softmax-Ersatzgradient. Experimente zu strukturierten Ausgaben, generativen Modellen und semi-supervidierter Klassifikation. Temperatur verändert Approximationsfehler und Gradientenvarianz. | Direkter methodischer Kandidat für unseren kategorischen Sampler. Unser jetziger Ersatzgradient ist nicht automatisch derselbe. Weiches Training garantiert keine korrekten harten Ausgaben; diese müssen während und nach dem Training geprüft werden. |
| **Yin et al.: Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets, ICLR 2019.** [Original](https://arxiv.org/abs/1903.05662) | Unter einem eingeschränkten Modell mit binären Aktivierungen und gaußschen Eingaben korreliert ein geeigneter Ersatzgradient mit einer Abstiegsrichtung; ungeeignete Varianten können instabil sein. | Begründet die Prüfung des Ersatzgradienten. Kein Konvergenzbeweis für unseren rekurrenten kategorischen Kern. Ein kleiner enumerierbarer Sampler-Test könnte Approximation und tatsächliche Verbesserung vergleichen. |
| **Hafner et al.: Mastering diverse control tasks through world models, Nature 2025 — DreamerV3.** [Original, World model learning](https://www.nature.com/articles/s41586-025-08744-2) | Erfolgreicher World Model mit rekurrentem und kategorischem Zustand. Gemeinsame Beobachtungs-/Reward-/Fortsetzungsprädiktion; getrennt gewichtete Dynamik-/Repräsentations-KL, Free Bits und 1 % Uniform-Mischung stabilisieren das Lernen. | Zeigt, dass kategorische Zustände grundsätzlich brauchbar sind. Wir haben bereits 1 % Uniform-Mischung; Dreamers gesamte Trainingsaufgabe fehlt in diesem kleinen Klassifikationstest. Free Bits können hier keinen nicht vorhandenen KL-Druck beseitigen. Kein Beleg, dass jede Größe oder jedes Lernziel genügt. |
| **Zhao et al.: Continuous First, Discrete Later: VQ-VAEs Without Dimensional Collapse, arXiv v2, Mai 2026.** [Original, §3–5](https://arxiv.org/html/2605.06870v2) | Preprint, hier keine begutachtete Veröffentlichung bestätigt. Unquantisiertes Autoencoder-Vortraining vor VQ verbessert im gleichen Gesamtbudget Bild- und Audio-Codecs; VQGAN-rFID sinkt in den berichteten Vergleichen um 17–35 %. Die Theorie verwendet lineare Encoder/Decoder und unabhängige gaußsche Faktoren. | Motiviert eine echte kontinuierliche Lernphase vor der Diskretisierung. VQ-Geometrie und Commitment-Verlust unterscheiden sich von unserem Sampler: Analogie, keine direkte Reparaturgarantie. Unser bisheriger Temperatur-Warmup zog weiterhin harte Codes und testete diese Methode daher nicht. |
| **Pezeshki et al.: Gradient Starvation: A Learning Proclivity in Neural Networks, NeurIPS 2021.** [Original, §3.4](https://papers.nips.cc/paper/2021/file/0987b8b338d6c90bbedd8631bc499221-Paper.pdf) | Analysiert unter Kreuzentropie konkurrierende prädiktive Merkmale. Spectral Decoupling bestraft die Klassifikationslogits und kann die Dominanz einzelner Merkmale verringern; geprüft an Klassifikations-/Robustheitsaufgaben. | Plausible Problemfamilie, aber hier eigene Zielköpfe: Farbe kann den Richtungsverlust nicht allein erfüllen. Deshalb nicht ohne Messung „Gradient Starvation“ diagnostizieren. Logit-Strafe ist außerdem nicht dasselbe wie Normalisierung der latenten Rohlogits. |
| **Yu et al.: Gradient Surgery for Multi-Task Learning, NeurIPS 2020 — PCGrad.** [Original](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | Projiziert Gradienten bei widersprüchlichen Richtungen zwischen Aufgaben. Verbesserungen auf untersuchten überwachten Multi-Task- und RL-Aufgaben. | Passt strukturell zu mehreren Verlusten auf gemeinsamen Parametern. Zuerst Normen und Winkel der Faktor-/Modalitätsgradienten messen und Einzelfaktortraining prüfen. Kein Ersatz für fehlende Information. |
| **Xin et al.: Do Current Multi-Task Optimization Methods in Deep Learning Even Help?, NeurIPS 2022.** [Original](https://papers.nips.cc/paper_files/paper/2022/hash/580c4ec4738ff61d5862a122cdf139b6-Abstract-Conference.html) | Auf den untersuchten Sprach-/Bildaufgaben erreichen spezialisierte Multi-Task-Optimierer keine Vorteile gegenüber sorgfältig eingestellten traditionellen Verfahren. | Gegenbeleg gegen automatische PCGrad-Adoption: einfache Verlustgewichtung, Lernrate und ein vergleichbares Suchbudget gehören in den Vergleich. Widerlegt nicht jeden möglichen Nutzen von PCGrad. |
| **Locatello et al.: Weakly-Supervised Disentanglement Without Compromises, ICML 2020.** [Original und Supplement](https://proceedings.mlr.press/v119/locatello20a.html) | Beobachtungspaare mit teilweise gemeinsamen Faktoren ermöglichen unter bestimmten Annahmen identifizierbare Faktoren; praktische Verfahren zeigen Vorteile auf mehreren Disentanglement-Benchmarks und Folgeaufgaben. | Gute Grundlage für kontrollierte Paare: gleiche Farbe/Position, andere Richtung. Ein Paar allein ist noch kein Lernziel. Wir müssen die gewünschte Unterscheidung in einem konkreten Verlust ausdrücken; die Identifizierbarkeitstheorie überträgt sich nicht automatisch auf diskrete rekurrente Zustände. |
| **Lee et al.: Set Transformer, ICML 2019.** [Original, §3.2](https://proceedings.mlr.press/v97/lee19d.html) | Attention-Pooling mit mehreren gelernten Seed-Queries liefert mehrere Ausgabevektoren; Interaktionen zwischen ihnen werden weiter verarbeitet. Geprüft auf Mengenaufgaben, unter anderem Clustering. | Kleine austauschbare Alternative zur sofortigen Query-Mittelung: getrennte Vektoren bis zur Zustandsprojektion erhalten. Keine Garantie, dass diese „Farbe/Ort/Richtung“ bedeuten. Permutationsinvarianz über Datensätze/Objekte ist nicht dasselbe wie Invarianz gegen das Spiegeln eines Bildes. |
| **Cohen und Welling: Group Equivariant Convolutional Networks, ICML 2016.** [Original](https://proceedings.mlr.press/v48/cohenc16.html) | Nutzt Symmetrien durch geteilte Gewichte; Ergebnisse auf rotiertem MNIST und CIFAR-10. | Für Orientierung ist Äquivarianz sinnvoll: Ein gespiegeltes Bild soll eine passend veränderte Repräsentation liefern. Vollständige Spiegelungsinvarianz wäre für Ost/West falsch. Optional für spätere Bildtests; erklärt nicht, warum explizite Richtungswörter im Kern versagen. |
| **Kipf, van der Pol, Welling: Contrastive Learning of Structured World Models, ICLR 2020 — C-SWM.** [Original, §2](https://arxiv.org/abs/1911.12247) | Objektweise Latents, relationale Dynamik und kontrastive Zustands-Aktions-Übergänge. Ergebnisse auf strukturierten Objektwelten, einfachen Atari-Szenen und Physiksimulationen; keine allgemeine Weltmodellgarantie. | Passt zu unserem längerfristigen Entity-/Relationsdesign. Eine Zukunftsvorhersage kann Bewegungsrichtung relevant machen. Das ist eine zusätzliche Trainingsaufgabe; reines Speichern eines Graphen erzeugt diese Fähigkeit nicht. |
| **Kipf et al.: Conditional Object-Centric Learning from Video, ICLR 2022 — SAVi.** [Original](https://arxiv.org/abs/2111.12594) | Sequenzielle Slot Attention; Optical-Flow-Ziel und anfängliche Positionshinweise verbessern Segmentierung/Tracking in realistisch gerenderten synthetischen Videos, auch unter untersuchten Verteilungsänderungen. | Konkrete spätere Lösungsklasse für Objektbewegung und Binding. Kein Beleg für Webcam-Lernen ohne weitere Daten/Ziele. Für das jetzige binäre Richtungsproblem wäre die vollständige Integration ein großer Schritt. |

## Abgeleitete Vergleichsreihe — Vorschlag, noch nicht ausgeführt

Die folgende Reihenfolge ist unsere Schlussfolgerung aus Literatur und lokalen
Diagnosen, kein von einer einzelnen Arbeit übernommener Erfolgsplan.

1. **Lernkonflikt isolieren.** Verifizierten Encoder einfrieren. Gleiche Architektur
   von frischer Initialisierung einmal nur Richtung, einmal alle Faktoren lernen
   lassen. Faktorgradienten am gemeinsamen Updater nach Norm und Kosinus vergleichen.
   Ein positiver Einzelfaktorversuch macht gemeinsame Optimierung zum gezielten
   Folgeproblem; ein Fehlschlag hält Repräsentation/Sampling/Optimierung offen.

2. **Die diskrete Schnittstelle kontrollieren.** Vorübergehend kontinuierliche
   Merkmale als Diagnose trainieren; daneben unveränderte harte Codes sowie eine
   Gumbel-Softmax-/Straight-Through-Variante. Ein kontinuierlicher Vorteil ist kein
   Vergleich bei gleicher Informationsrate. Danach innerhalb eines festen Gesamtbudgets
   eine kontinuierliche Startphase plus harte Endphase gegen durchgehend harte
   Ausbildung prüfen. Abschlussmessung ausschließlich im gewünschten Laufzeitpfad,
   mit wiederholten Stichproben. Keine Hilfsausgabe oder Zielinformation am Decoder.

3. **Erst den gemessenen Engpass verändern.** Wenn Gradienten konfligieren, PCGrad
   gegen gut eingestellte Faktor-Gewichte vergleichen. Wenn das Mittel der Query-Tokens
   schwächer bleibt, getrennte Queries bzw. Attention-Pooling vor derselben kategorischen
   Ausgabe prüfen. Nicht beide Änderungen gleichzeitig einführen. Parameter, tatsächliche
   Rechenzeit und effektive Codenutzung mitberichten.

4. **Zusammensetzung prüfen.** Paare mit nur einer Faktoränderung erlauben einen
   expliziten Kontrast im Richtungsleser des tatsächlich gesampelten Zustands; stabile
   Faktoren müssen zugleich richtig bleiben. Eine bloße Richtungshöhergewichtung mit
   identischem Paar-Sampling ist ein notwendiger Kontrollversuch. Wir schreiben keine
   dauerhaft benannten latenten Dimensionen vor. Danach unabhängige neue Kombinationen,
   einzelne Modalitäten, vollständige und komplementäre Eingaben prüfen.

Bei allen Varianten: globale Codevielfalt reicht als Qualitätsmetrik nicht. Zusätzlich
Richtungsverwechslungen **bei gleicher Farbe und Position**, Genauigkeit vor/nach
Sampling, Retention der anderen Faktoren und Fehlerraten unter wiederholten Draws
messen. Eine Code-Marginalverteilung kann vielfältig aussehen und trotzdem eine
relevante Unterscheidung ignorieren. Unser bisheriger Entwicklungs-Holdout ist mehrfach
betrachtet; eine belastbare Generalisierungsbehauptung braucht neue unberührte Fälle.

Eine gerichtete Zustandsänderung, Objekt-Slots oder Äquivarianz werden erst ergänzt,
wenn eine entsprechende Aufgabe und Diagnose sie rechtfertigen. Für spätere reale
Bewegung sind C-SWM und SAVi relevante Ausgangspunkte; für die jetzige Reparatur sind
sie keine Voraussetzung. Ein Bedarf für einen großen Sprachdecoder folgt aus keiner
der hier geprüften Arbeiten.

## Review und Grenzen

Claude erhielt ausschließlich öffentliche Paper-Zusammenfassungen und Transferfragen;
keinen Projektcode, keine Daten und keine Messwerte. Die methodischen Grenzen wurden
unabhängig geprüft. Korrekturen: Straight-Through Gumbel-Softmax steht bereits im
Original von 2017; Commitment, KL und Spectral Decoupling sind unterschiedliche Ziele.
Eine kontinuierliche Startphase und ein Warmup mit weiterhin harten Stichproben sind
nicht derselbe Eingriff. Details und Reconciliation unter
`runs/reviews/direction_literature_v1/`.

Die Literatur liefert begründete Kandidaten, keine Diagnosebestätigung. Dokumentation
und Quellenvergleich wurden ergänzt; Modellgewichte, Standardarchitektur und bisherige
Validierungsfarben bleiben unverändert.
