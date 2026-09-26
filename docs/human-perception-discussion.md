# Menschen: Pose, Segmentierung und zeitlicher Zustand

22. September 2026. Alex fragt, ob nach Personenerkennung automatisch ein Skelett
geschätzt und die Person segmentiert werden sollte, damit der Agent Pose und Shape
nutzen und darauf reagieren kann. Dies ist eine offene Architekturfrage; die folgende
Einordnung ist ein Vorschlag, keine angenommene Implementierung oder Validierung.

Pose, sichtbare Silhouette und Körperform liefern unterschiedliche Informationen:

| Darstellung | Inhalt | Grenze |
| --- | --- | --- |
| Gelenkpunkte / Skelett | Haltung, Gelenkpositionen, über Zeit Bewegung | Keine vollständige Oberfläche, Kleidung oder sichere Handlungsabsicht |
| Instanzmaske | Sichtbare Pixel einer konkreten Person, Umriss und Bildfläche | Verdeckte Teile fehlen; Kleidung verändert den Umriss |
| Parametrisches 3D-Körpermodell | Geschätzte Pose und Körperproportionen | Monokulare Tiefe, Maßstab und verdeckte Form bleiben mehrdeutig |
| Bildmerkmale und Umgebung | Erscheinung, Objekte, feine visuelle Hinweise | Müssen mit Person, Zeit und Koordinaten verknüpft bleiben |

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) schätzt
Gelenkpunkte; seine dokumentierte 3D-Rekonstruktion trianguliert mehrere Ansichten.
Ein gezeichnetes Skelett ist noch kein angepasstes Oberflächenmodell.
[MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
liefert 33 Landmarken, geschätzte 3D-Koordinaten und optional Segmentierungsmasken.
Es ist ein Kandidat für eine erste Machbarkeitsprobe, kein hier gemessener Gewinner.
[SMPL-X / SMPLify-X](https://smpl-x.is.tue.mpg.de/) modelliert beziehungsweise schätzt
3D-Körper, Hände und Gesicht. Eine solche zusätzliche Formschätzung lohnt sich als
spätere Hypothese, wenn eine konkrete Aufgabe 3D-Geometrie benötigt.

Für PATH-WM passt die Idee zu spezialisierten Verbrauchern der akzeptierten
[gemeinsamen Multiskalenrepräsentation](multiscale-modality-design.md): Detektion,
Pose und Instanzsegmentierung lesen passende Skalen und liefern ergänzende
abgeleitete Beobachtungen. Ein externes vortrainiertes Modell wäre zunächst ein
separater Beobachtungsadapter oder Lehrer; es teilt nicht automatisch unsere
Encoder-Gewichte. Seine Ausgaben sind Pseudolabels, keine unabhängige Wahrheit.

Ein vorgeschlagener Personenbezug verbindet Bildmerkmale, Maske, Gelenke,
Zeitstempel, Koordinaten-/Kamerabezug, Sichtbarkeit und Modellherkunft. In Video
kommt eine lokale Track-ID mit unsicherer Zuordnung hinzu. Bei Kreuzung oder
Verdeckung keine erzwungene Identitätsverschmelzung; kurze Vorhersagen bleiben
als solche markiert. Online werden nur bis zur Entscheidung verfügbare Frames
verwendet. Relative Körperkoordinaten sind keine gemessene globale Raumposition.

Bild-/Körperform ändert sich anders als Pose: Proportionen können innerhalb eines
Tracks langsam aktualisiert werden, Pose und Bewegung häufiger; Kamerabewegung,
Perspektive und Kleidungswechsel müssen berücksichtigt werden. Ein harter
Detektionsschwellwert darf keine dauerhafte Blindstelle erzeugen: erneute Suche
und allgemeine Bildmerkmale bleiben verfügbar. Eine gemeinsame Personenmaske
aller Menschen reicht für mehrere getrennte Personen nicht aus.

Für eine Reaktion folgt auf die Wahrnehmung eine eigene Verarbeitung:
zeitlicher Personenzustand und Objektbeziehungen → Handlungshypothese →
aufgabenabhängige Policy → Ausführung und beobachtete Wirkung. Etwa stützen eine
Handbewegung zur Tasse und anschließende gemeinsame Bewegung die Hypothese
„greift nach der Tasse“. Nähe allein beweist keinen Kontakt; Pose allein keine
Absicht. Siehe [Aktionssemantik](action-semantics-design.md). Allgemeine
Personenerkennung aus Pixeln und diese Reaktionskette sind durch die bisherigen
Experimente mit vorgegebenen Entity-Deskriptoren nicht validiert.

Angewandte Designprinzipien: gemeinsame Quellen mehrfach nutzen; feine und grobe
Skalen zugänglich halten; Spezialisten nach Bedarf aufrufen; Quellbilder und
abgeleitete, revidierbare Schätzungen getrennt halten; langsam und schnell
veränderliche Zustände unterschiedlich aktualisieren. Masken/Skelette komprimieren
verlustbehaftet und ersetzen daher nicht die verfügbaren Bildmerkmale.
Zusätzliche Köpfe, Maskenspeicher und Personenanzahl kosten Parameter, Rechenzeit
und Speicherverkehr; bedingte Aufrufe sparen nur unter gemessenem Routingaufwand.

Kleinster sinnvoller Vergleich vor einer Übernahme: dieselben nach Person/Video
getrennten Sequenzen, identische Aufgaben und Bildbasis; Bildmerkmale allein,
+ Pose, + Instanzmaske, + beide. Erst eine Aufgabe wie Winken versus Armheben oder
Hand-Tasse-Interaktion festlegen, dann Daten, Schwellen, Seeds und Budget vorab
bestimmen. Neben Pose-/Maskenfehlern zählen falsche Reaktionen, verpasste Ereignisse,
Track-Wechsel, Reaktionsverzögerung sowie gesamte Latenz/Arbeit/Speicher. Verdeckung,
mehrere Personen und fehlgeschlagene Detektion gehören in die Auswertung. Nutzen
für Agentenentscheidungen bleibt eine zu prüfende Hypothese; kein Lauf gestartet.

## Nachtrag: vollständige 3D-Körperform und SKEL

Alex fragt anschließend konkret nach [SKEL](https://skel.is.tue.mpg.de/).
SKEL ist ein differenzierbares parametrisches Körpermodell: Formparameter und
biomechanische Poseparameter erzeugen Körperoberfläche, Gelenke und ein inneres
Skelettmesh. Die [offizielle Implementierung](https://github.com/MarilynKeller/SKEL)
dokumentiert zehn Formparameter und 46 Poseparameter. Die Formbasis stammt von
SMPL; SKEL verändert insbesondere die Gelenkstruktur und zulässige Artikulation.
Dies ist keine Messung der individuellen verdeckten Knochengeometrie.

Die Vorwärtsrichtung lautet schematisch `Form + Pose → 3D-Mesh`. Um Bilder zu
verstehen, braucht man die umgekehrte Schätzung von Form, Pose und Kamera bzw.
globaler Platzierung. Zwei Wege sind zu unterscheiden:

1. Ein trainierter Bildregressor schätzt zuerst SMPL-Parameter. Beispielsweise
   [HMR 2.0 / 4DHumans](https://shubham-goel.github.io/4dhumans/) liefert
   bildbasierte Körperrekonstruktion und ein darauf aufbauendes Trackingverfahren.
   Danach passt man SKEL an die SMPL-Meshes an. SKEL stellt dafür
   [Frame-](https://github.com/MarilynKeller/SKEL/blob/master/examples/align_to_SMPL_frame.py)
   und Sequenzbeispiele bereit. Die Kombination ist hier ein Integrationsvorschlag,
   kein installierter oder geprüfter PATH-WM-Pfad.
2. Alternativ könnte ein eigener Schätzer SKEL-Parameter direkt vorhersagen oder
   diese anhand projizierter Gelenke und gerenderter Silhouetten optimieren.
   Differenzierbarkeit ermöglicht diese Konstruktion; ein fertiger allgemeiner
   RGB-zu-SKEL-Schätzer ist durch die verlinkte Veröffentlichung nicht belegt.
   [SMPLify](https://smplify.is.tue.mpg.de/) demonstriert das Prinzip, ein
   statistisches 3D-Körpermodell an erkannte 2D-Gelenke anzupassen.

Als vorgeschlagenes Anpassungsziel dienen Gelenk-Reprojektionsfehler,
sichtbarkeitsabhängige Silhouettenfehler, Form-/Poseprior und zeitliche Konsistenz;
bei verfügbarer kalibrierter Tiefe zusätzlich ein Tiefenfehler. Eine bekleidete
Personenmaske darf nicht naiv exakt mit der modellierten Körperhaut gleichgesetzt
werden. Verdeckungen durch Objekte müssen beim Rendervergleich berücksichtigt
werden. Gelenkpunkte allein legen Körperumfang oder Tiefe nicht eindeutig fest.

Vollständig heißt hier: Das Modell liefert eine gesamte Oberfläche einschließlich
verdeckter Seiten. Deren Richtigkeit ist nicht durch ihre Existenz belegt. Neue
Blickrichtungen und kalibrierte Mehrkamerabilder/Tiefe können Mehrdeutigkeiten
verringern; eine monokulare RGB-Aufnahme bestimmt den absoluten Maßstab und
individuelle verdeckte Form nicht eindeutig. Körperoberfläche, Kleidung und Haare
sind unterschiedliche Rekonstruktionsziele; SKEL ist kein vollständiger bekleideter
Avatar einschließlich beliebiger Oberflächendetails.

Für Video wäre ein gemeinsamer, vorsichtig aktualisierter Formzustand pro Person
mit zeitabhängiger Pose, globaler Platzierung und Kamerazustand sinnvoll. Online
darf die Schätzung nur bereits verfügbare Bilder nutzen. Bei Unsicherheit bleiben
mehrere Hypothesen oder ein als unsicher markierter Fit bestehen. Ein gutes
Reprojektionsbild beweist noch keine richtige 3D-Geometrie.

Für PATH-WM wäre SKEL damit ein Kandidat für eine strukturierte Personenhypothese:
kompakte Form-/Poseparameter plus Unsicherheit und Quellenbezug; bei Bedarf wird
das Mesh abgeleitet. Bildmerkmale, sichtbare Maske und Objektbeziehungen bleiben
zugänglich. Ein kinematisches Körpermodell allein liefert keine Handlungsabsicht,
Kontaktkräfte oder gelernte zukünftige Dynamik. Vor einer Übernahme ist gegen
einfachere 2D-/3D-Gelenkzustände zu prüfen, ob SKEL Agentenreaktionen verbessert
und welchen Zusatzaufwand Regression, Anpassung und Mesh-Erzeugung verursachen.

## Neuere Verfahren: Recherche am 22. September 2026

Alex fragt nach neuerem Stand der Technik, besserer Genauigkeit und Geschwindigkeit.
Die bisherige SKEL-Einordnung bezog sich auf die originale Modellveröffentlichung.
Die erweiterte Suche findet bereits veröffentlichte direkte Bild-zu-SKEL-Schätzer:
HSMR und SKEL-CF. Dafür muss also nicht erst ein eigener Bildschätzer entwickelt
werden. Die SMPL-Zwischenstufe bleibt eine Option, ist aber keine Voraussetzung.

| Kandidat | Rolle und datierter Stand | Aussage und Grenze der Primärquellen |
| --- | --- | --- |
| [SKEL-CF](https://pokerman8.github.io/SKEL-CF/) | Direkte RGB-zu-SKEL-Schätzung; Code/Gewichte seit November 2025, ECCV 2026 | MOYO: 85.0 mm MPJPE gegenüber HSMR 104.5 mm im Autorenvergleich. Passender Kandidat für explizite biomechanische Pose plus Körpermesh. Keine belastbare lokale Laufzeit; ViTPose-H ist kein kleiner Backbone. |
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) | Einzelbild-Rekonstruktion mit MHR, Körper/Hände/Füße; Gewichte November 2025, Paper Februar 2026 | Unterstützt Masken-/Keypoint-Prompts. Offizieller DINOv3-H+-Backbone mit 840M Parametern; 3DPW 54.8 mm, EMDB 61.7 mm MPJPE in veröffentlichten Protokollen. Kein Vergleich dieser Werte mit MOYO und kein Beweis genauer individueller Körperumfänge. |
| [Fast SAM 3D Body](https://arxiv.org/html/2603.15603v1) | Beschleunigte SAM-3D-Body-Verarbeitung, März 2026 | Tabelle 1, automatische Detektion, RTX 6000 Ada, Batch 1: auf 3DPW 0.8 → 6.6 Frames/s; MPJPE 57.6 → 58.9 mm. Separate Teleoperationsdemo: etwa 65 ms auf RTX 5090. Beschleunigung umfasst Approximationen; kein universeller Genauigkeitsgewinn. |
| [Human3R: Everyone Everywhere All at Once](https://fanegg.github.io/Human3R/) | Online-Rekonstruktion mehrerer Menschen, Szene und Kamera; Oktober 2025 / ICLR 2026 | SMPL-X, kausaler fortlaufender Zustand. Autoren berichten 15 FPS mit kleinerem ViT-S-Prior bei 672 Pixeln sowie ca. 8 GB Speicher; diese Angaben begründen keine RTX-3050-Leistung. Schwerpunkt gemeinsamer Raum-/Bewegungskontext. |
| [DETRAM](https://research.nvidia.com/labs/amri/publication/lee2026detram/) | Gemeinsame Detektion, Tracking und Mesh-Schätzung; Juli-Preprint / ECCV September 2026 | Persistente Identitätsqueries; Autoren berichten führende Trackingwerte und konkurrenzfähige Rekonstruktion. In dieser Sichtung keine verifizierte lokale Laufzeit oder einsatzbereite Installation. Aktueller Forschungsvergleich, kein pauschaler Genauigkeitssieger. |

Für SKEL-CF wurden offizielle Projektseite, Code-README und Paper gelesen; die
README verweist auf freigegebene Gewichte. SAM-3D-Body- und Human3R-Paper wurden
über Hugging Face gelesen. Dessen Markdown zeigte bei SKEL-CF v3 und Human3R v1;
aktuellere Projektseiten ergänzen Veröffentlichungsstatus, aber spätere Paperrevisionen
wurden nicht vollständig abgeglichen. Fast-SAM-Paper v1 wurde nach einem
Hugging-Face-404 direkt auf arXiv geprüft. Insbesondere die Hardware, Oracle- versus
Automatic-Protokolle und Genauigkeitsverluste stammen aus der Primärquelle;
sekundäre 4090-/4060-Laufzeiten wurden nicht übernommen.

MPJPE misst Gelenkpositionsfehler; Oberflächenfehler, Körpermaße, Hände,
Track-Stabilität und Reaktionsqualität sind eigene Ziele. Eine bessere Rangposition
für Gelenke oder Tracking beweist keine genauere individuelle Körperform.
Ein [unabhängiger SAM-3D-Body-Preprint](https://arxiv.org/abs/2601.06035)
berichtet Grenzen bei individuellen Formabweichungen; seine Erklärung ist eine
Autorenhypothese, kein hier isolierter Kausalnachweis. Verdeckte Geometrie bleibt
auch bei neueren Verfahren modellabhängig geschätzt.

Vorgeschlagene Priorisierung für PATH-WM: SKEL-CF als Referenz für die konkrete
biomechanische Repräsentation, SAM 3D Body / Fast SAM 3D Body für den Vergleich
von Robustheit und Laufzeit, Human3R für gemeinsam geschätzten Menschen-/Szenenraum.
Große Modelle könnten zunächst Lehrer oder bedingt aufgerufene Spezialisten sein;
kleine kausale Leser nutzen gemeinsam vorbereitete Merkmale. Lehrerlabels behalten
Fehler und müssen unabhängig geprüft werden. Die Projekt-GPU RTX 3050 8 GiB
erfordert eigene Messungen einschließlich Detektion, Kamera, Personenanzahl und
Agentenanteil; nicht aus größeren GPUs oder nominell 8 GB Speicher extrapolieren.
Keine Installation, Gewichtsdownloads, Training, lokale Benchmarks oder Adoption.

## Hände, Füße und einzelne Zehen

22. September, weiterer Nutzerfokus: Welche Verfahren bekommen Hände und Füße
wirklich richtig hin? Eine vollständige Körperoberfläche allein belegt keine
korrekte Finger-/Zehenartikulation oder Objekt-/Bodenkontakte.

- **Hände/Finger:** [WiLoR](https://github.com/rolpotamias/WiLoR), CVPR 2025,
  lokalisiert Hände und rekonstruiert artikulierte MANO-Handmeshes aus Ausschnitten.
  Das [Paper](https://arxiv.org/abs/2409.12259) berichtet auf FreiHAND 5.5 mm
  PA-MPJPE und auf HO3Dv2 7.5 mm. Die nachträgliche Procrustes-Ausrichtung entfernt
  globale Ähnlichkeitstransformationen; diese Werte sind keine Garantie für
  millimetergenaue Fingerpositionen im gemeinsamen Szenenraum. Detektor-FPS sind
  nicht die Laufzeit des vollständigen 3D-Rekonstruktors. März-2026-Code bietet
  einen optionalen Fast-Modus; hier nicht ausgeführt.
- **Fußausrichtung in Videos:** [FootMR](https://twehrbein.github.io/footmr-website/),
  3DV 2026, verbessert die Knöchelrotationen bestehender SMPL-X-Bewegungsschätzer
  mit 2D-Fußsequenzen und Knie-/Knöchelkontext. Autoren berichten bis zu 30 Prozent
  weniger Knöchelwinkelfehler auf MOYO gegenüber der besten verglichenen
  Videomethode. [Abschnitt 5](https://arxiv.org/html/2603.09681v1#S5) benennt
  ausdrücklich, dass die vereinfachten SMPL-X-Füße Zehenkrümmen nicht darstellen.
  Das Modell nutzt ein 120-Frame-Aufmerksamkeitsfenster; daraus folgt weder
  automatisch kausale Online-Verarbeitung noch vier Sekunden Ausgabelatenz.
- **Einzelne Zehen und Fußdeformation:** [SUPR-Foot](https://supr.is.tue.mpg.de/)
  besitzt eine differenziertere Fußartikulation und gelernte bodenkontaktabhängige
  Verformungen. Es ist ein parametrisches Modell, kein allein ausreichender
  RGB-zu-Zehen-Bewegungsschätzer. FootMR nennt es als künftige Erweiterung mit
  zusätzlichen benötigten 2D-Landmarken; diese Kombination ist nicht veröffentlicht
  validiert durch den FootMR-Nachweis.
- **Detaillierte Fußoberfläche:** [FOCUS](https://github.com/OllieBoyne/FOCUS),
  3DV 2025, rekonstruiert aus mehreren Ansichten mit dichten Korrespondenzen,
  optional über FIND-Modellanpassung. Dieser Aufnahme-/Rekonstruktionsvertrag
  unterscheidet sich vom kontinuierlichen Verfolgen aller Zehen in beliebigem
  Ganzkörpervideo.

SAM 3D Body bleibt ein sinnvoller gemeinsamer Vergleichskandidat. Die
[offizielle Beschreibung](https://ai.meta.com/blog/sam-3d/) benennt ausdrücklich,
dass die Handgenauigkeit spezialisierte Handmodelle nicht übertrifft. Der
[MHR70-Ausgabevertrag](https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/metadata/mhr70.py)
enthält für jeden Fuß Ferse, großen und kleinen Zeh zusätzlich zum Körper-Knöchel;
das ist keine Messung sämtlicher Zehengelenke. Daraus wird hier keine Aussage
abgeleitet, dass die gesamte interne MHR-Repräsentation nur diese Punkte hätte.

Vorschlag für PATH-WM: Körperkontext und Personenbezug gemeinsam verarbeiten,
bei benötigtem Detail Originalbild-Ausschnitte bzw. feine gemeinsame Skalen für
Hände/Füße lesen und die lokalen Ergebnisse in denselben Koordinatenraum und
Zeitbezug zurückführen. Vergrößern eines bereits informationsarmen Ausschnitts
erzeugt keine beobachteten Fingerdetails. Kontakt und Verdeckung brauchen
Objekt-/Bodenevidenz; eine plausible Greifpose allein bestätigt keinen Kontakt.
Der Kandidat nutzt gemeinsame Quellen, mehrere Auflösungen und bedingte
Spezialisten. Dafür fallen Crop-/Lesekosten, zusätzliche Inferenz und zeitliche
Verzögerung an. Kleinster Vergleich: gemeinsamer Körperleser versus zusätzlicher
Detailpfad, identische Videos/Personen und Budgets; Finger-/Fußfehler, falsche
Kontakt-/Greifentscheidungen und gesamte Latenz getrennt messen. Keine neue
Implementierung, Adoption oder Validierung.

## Gesicht: Form, Mund, Augen und Ausdruck

22. September, Folgefrage nach Mund, Augen, Nase und Gesichtsausdruck. Vorschlag:
einen Gesichtsleser an denselben Personenbezug anbinden; stabile Gesichtsform,
Kopfpose, zeitabhängige Artikulation und interpretierte soziale Signale trennen.
Geometrische Formparameter identifizieren dabei keine namentlich bekannte Person.

| Ziel | Erfasste bzw. vorgeschlagene Repräsentation | Kandidat / Grenze |
| --- | --- | --- |
| Gesichtspunkte und sichtbare Mimik | Lippen-/Lidkonturen, Nase, Brauen, Iris-Landmarken, Ausdruckskoeffizienten | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker): 478 geschätzte 3D-Landmarken, 52 Blendshape-Scores, optional Transformationsmatrizen; keine exakte individuelle metrische Kopfmessung. |
| Ausdrucksstarkes 3D-Gesicht | Kopf-/Kieferpose, Form, Ausdruck und Lidschluss | [SMIRK](https://github.com/georgeretsi/smirk), CVPR 2024, auf FLAME-Basis mit Ausrichtung auf asymmetrische/extreme Ausdrücke und Mund-/Augenschluss. [Paper](https://arxiv.org/abs/2404.04104) geprüft; kein aktueller universeller Genauigkeitssieger behauptet. |
| Parametrischer Kopf | Gesichtsform, Kiefer, Hals, Augäpfel und Ausdrucksbasis | [FLAME](https://flame.is.tue.mpg.de/) ist das darstellende Modell; die Bildschätzung ist ein eigener Teil. Modellversion und Erweiterungen bestimmen verfügbare Lid-/Munddetails. |
| Blickrichtung | Geschätzte Blickwinkel bzw. Richtung mit Koordinatenbezug | [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) als eigenständiger Vergleichskandidat; Irisposition und Kopfrotation allein garantieren keinen präzisen Blickpunkt. Ein konkretes Zielobjekt braucht zusätzlich Szenengeometrie und gegebenenfalls Kalibrierung. |
| Gemeinsamer Körper/Kopf | Ausdrucksstarke Körper-/Gesichtsparameter in einem Mesh | [SMFLIX](https://www.mdpi.com/2227-7390/14/18/3287), 10. September 2026: integriert SMPL-X und FLAME samt Lid-/Mundkomponenten. Neuere Autorenresultate, Code/Modell laut Artikel auf Anfrage; keine lokale Verfügbarkeit oder Überlegenheit bestätigt. |

Primärquellen bestätigen MediaPipes Bild-/Video-/Live-Modi. Die
[Blendshape-Liste](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/drawing_styles/face_landmarker/Blendshapes)
enthält unter anderem `eyeBlinkLeft`, `jawOpen`, `mouthSmileLeft` und
`noseSneerLeft` in entsprechender API-Schreibweise. Das sind Koeffizienten für
sichtbare Deformationen, keine kalibrierten Wahrscheinlichkeiten innerer Emotionen.
Für mehrere Gesichter ist Zuordnung gesondert zu prüfen; laut Dokumentation gilt
die eingebaute Glättung nur für `num_faces=1`.

SMIRK-Papertext über Hugging Face (v2) und offizielle Repositories gelesen.
SMFLIX-Verlagstext kam aus dem Suchindex; erneutes direktes Öffnen ergab HTTP 429,
die offizielle Projektseite war lesbar. Berichtete 48 FPS auf RTX 5090 sind die
Netzrate; 21 FPS bezeichnet die vollständige Webcam-Kette. Keine Übertragung auf
RTX 3050, keine eigene Messung. Der Artikel benennt Blick-/Zungenmodellierung als
offene Arbeit. Ein dichtes Gesichtsmesh bedeutet generell nicht, dass Zähne,
Zunge, Mundinnenraum oder die Blickachse korrekt aus dem Bild rekonstruiert sind.

Für PATH-WM zuerst MediaPipe als günstige externe Beobachtungsreferenz gegen
einen detaillierteren Gesichtsleser wie SMIRK vergleichen. Spätere gemeinsame
Multiskalenleser können entsprechende Köpfe lernen. Die langsamer veränderliche
Form kann pro Track zusammengeführt werden; Kopfpose, Lider, Lippen und Brauen
werden zeitlich aktualisiert. Mehrere Auflösungen und gezielter Detailzugriff
entsprechen den stehenden Designprinzipien. Hochauflösende Quellmerkmale,
Zeitstempel, Sichtbarkeit und Unsicherheit bleiben erhalten; Glättung darf kurze
Blinzel-/Lippenereignisse nicht unbemerkt entfernen und ihre Verzögerung muss
gemessen werden.

Mimik bleibt zunächst ein beobachtbarer Bewegungsverlauf: Mundwinkel angehoben,
Lider geschlossen, Kopf gedreht. Emotion, Aufmerksamkeit oder Absicht sind
separate kontextabhängige Hypothesen. Sichtbare Lippenbewegung beweist noch kein
Sprechen; zur Sprecherzuordnung passen synchronisierte Audiosignale. Kleinster
Vergleich: identische gesichts-/videogetrennte Sequenzen mit unterschiedlicher
Auflösung, Kopfrotation und Verdeckung; Lippen-/Lidfehler, zeitliches Zittern,
verpasste Ereignisse und End-to-End-Latenz getrennt vom globalen Meshfehler.
Keine Implementierung, Adoption oder Validierung in dieser Diskussion.

## Personen wiedererkennen und das Gedächtnis erweitern

22. September, Sprachfolgefrage: Alex priorisiert zunächst Geschwindigkeit, dann
Genauigkeit, später Robustheit und fragt nach selbst gelerntem Wiedererkennen
einzelner Personen. Diese Reihenfolge ist Nutzerpräferenz; die folgende konkrete
Lösung bleibt ein Vorschlag. Keine Garantie einer exakten Identifikation.

Kurzfristige Track-ID, dauerhafter Personendatensatz und ein eventuell bekannter
Name sind getrennte Bezüge. Ein Track verfolgt eine Beobachtung über Frames;
Wiedererkennung verknüpft eine neue Beobachtung mit früherer Evidenz. Ein Name
braucht zusätzlich eine belegte Zuordnung. Pose, Kleidung, Mimik und geschätzte
3D-Körperform allein bilden keine verlässliche dauerhafte Identität.

Vorgeschlagener schneller Einstieg: ein gemeinsamer vortrainierter Merkmalsleser
erzeugt Gesichtsdeskriptoren; pro Person werden wenige hochwertige, verschieden
ausgerichtete Beispiele samt Quellenzeit, Qualität und Modellversion gespeichert.
[ArcFace](https://arxiv.org/abs/1801.07698) liefert das Prinzip unterscheidbarer
Gesichts-Embeddings; [InsightFace](https://github.com/deepinsight/insightface) ist
ein konkreter externer Vergleichskandidat. Das Modell wird nicht für jede neue
Person neu trainiert. [Deep SORT](https://github.com/nwojke/deep_sort) veranschaulicht
die Kombination von Bewegung und Erscheinungsdeskriptoren beim Tracking; dies
ist kein Nachweis lebenslanger Identität bei Kleidungs-/Ansichtswechseln.

Der gemeinsame Leser lernt allgemeine Unterscheidungsmerkmale; das schnelle
Lernen neuer Personen erfolgt zunächst durch gespeicherte Referenzen und deren
Vergleich. Für PATH-WMs gewünschte lernbare Filterbank kann später ein eigener
Identitätsleser über den gemeinsamen Multiskalenmerkmalen trainiert werden:
belegte gleiche Personen näher, verschiedene Personen weiter auseinander.
Ein vortrainierter externer Leser ist eine Baseline, keine Ablösung der akzeptierten
gemeinsamen Lernrichtung. Modellwechsel erfordern kompatible oder neu berechnete
Deskriptoren; alte und neue Merkmalsräume nicht unbemerkt mischen.

Neue Tracks oder gute neue Gesichtsansichten lösen den teureren Vergleich aus;
dazwischen läuft das günstigere Tracking. Ein Schwellenwert und Abstand zum
nächstbesten Kandidaten müssen an reservierten Daten kalibriert werden. Schlechte
Sicht bedeutet zunächst ungeklärt, nicht automatisch neue Person. Unsichere
Treffer dürfen keine alten Profile überschreiben; zusätzliche Referenzen erst
nach hinreichend belegter Zuordnung aufnehmen und Zuordnungen revidierbar halten.
Mehrere benachbarte Frames sind korrelierte Evidenz, keine unabhängigen Bestätigungen.

Angewandte Prinzipien: gemeinsame Merkmale wiederverwenden, langsam veränderliche
Identität von beweglichem Zustand trennen, kurze Track-Lebensdauer und dauerhafte
Evidenz explizit besitzen, Kandidaten günstig suchen und stärkere Prüfung gezielt
auslösen. Begrenzte Referenzzahl spart Speicher und Vergleichsarbeit, kann aber
seltene Ansichten verlieren. Kleinster Vergleich: gleiche zeitlich getrennte
Episoden bekannter und unbekannter Personen, feste Referenz-/Rechenbudgets,
Vergleich in jedem Frame versus ereignisabhängig; falsche Zuordnungen, verpasste
Wiedererkennung, ungeklärte Fälle, Track-Wechsel, Profilverunreinigung und gesamte
Latenz messen. Keine Implementierung, trainierten Personenprofile oder neue
Validierung. Die vorhandenen Experimente mit gelieferten Entity-Deskriptoren
validieren weiterhin keine Wiedererkennung aus natürlichen Personenbildern.

Direkte Nachfrage: Alex vermutet, dass bereits FLAME- bzw. SKEL-Parameter gut
für Wiedererkennung geeignet sind (Sprachtranskript „Lame / Scale“, aus dem
Gespräch als FLAME / SKEL interpretiert). Die Formparameter sind ein sinnvoller
zusätzlicher Personenhinweis: FLAME trennt Kopfform, Ausdruck und Artikulation;
SKEL trennt Körperform β und biomechanische Pose q. Quellen:
[FLAME](https://flame.is.tue.mpg.de/), [SKEL](https://skel.is.tue.mpg.de/).
Vorschlag: personenbezogene Form über mehrere gute Ansichten zusammenführen,
Pose/Mimik dagegen pro Zeitpunkt aktualisieren. Unterschiedliche Menschen können
ähnliche Formkoeffizienten haben; Bildschätzungen derselben Person können schwanken.
Eine Repräsentation mit separaten Parametern garantiert keine fehlerfreie Trennung
bei der inversen Bildschätzung. Daher formbasierte Kandidatensuche und Kombination
mit Erscheinungsmerkmalen prüfen, keine eindeutige Identität aus β voraussetzen.
Gleiche Modellbasis/Version, kamerabezogene Skalierung und Unsicherheit müssen
beim Vergleich kontrolliert werden. Das Wiederverwenden ohnehin berechneter Form
ist günstig; eine zusätzliche 3D-Schätzung allein für Identität ist gesondert zu
kosten. Kleinster Zusatzvergleich: Form allein, Gesichtsdeskriptor allein und
Kombination auf denselben zeitlich getrennten Episoden bei gleicher Zielrate
falscher Zuordnung. Nutzerhypothese aufgenommen, ohne Erfolgsbehauptung.

Alex ergänzt explizit separate Augen-/Munddetails. Vorschlag: aus denselben
hochauflösenden Quellen lokale Merkmale für Augenregion und Mund lernen und
zusammen mit Gesichts-/Körperform vergleichen. Personenabhängige Grundform und
zeitabhängige Lid-/Mundöffnung bzw. Mimik getrennt repräsentieren; diese Trennung
muss durch Daten/Training geprüft werden. Die geometrische Basis kann einen
kanonischen Bezug für lokale Merkmale liefern, ohne sichtbare Bilddetails in
den Formkoeffizienten vollständig abzubilden. Verdeckte oder unscharfe Bereiche
werden als fehlende bzw. schwache Evidenz behandelt. Die Signale stammen teilweise
aus denselben Pixeln: nicht als unabhängige Bestätigungen zählen. Zusätzliche
Augen-/Mundleser ereignisabhängig ausführen und Identitätsfehler bei Lachen,
Blinzeln, Sprechen, Profilansicht und ähnlichen Personen separat messen.
[MICA](https://wojciechzielonka.com/mica/) nutzt bereits Gesichtsidentitätsmerkmale
zur Schätzung metrischer Kopfform; dies stützt die Verbindung zwischen Aussehen
und Geometrie, validiert aber nicht den vorgeschlagenen kombinierten PATH-WM-Leser.
Der Nutzer schlägt zusätzliche Details vor, keine konkrete Implementierung oder
Modellwahl ist damit beschlossen.

## GPU-Budget für Personenwahrnehmung und zeitlichen Verlauf

22. September: Alex fragt, ob die besprochene Kombination auf seine GPU passt.
Im vorherigen Sprachverlauf schlug er zeitgestempelte Parameter unter der Entität
im Knowledge-Graph vor; hier wird deren Speicherort in die Budgetbetrachtung
aufgenommen. Noch keine komplette Pipeline ausgewählt oder profiliert.

Lokale Momentaufnahme mit `nvidia-smi`, 22. September 2026, 13:14 Ortszeit:
RTX 3050, 8192 MiB gesamt, 373 MiB reserviert, 993 MiB belegt, 6827 MiB frei.
`free -m`: 64137 MiB System-RAM, 53244 MiB verfügbar. Freier Speicher schwankt;
dies ist Hardware-/Belegungsprüfung, kein Inferenz- oder Trainingsbenchmark.

Die [bestehende Budgetskizze](multimodal-reference-design.md#gpu-arbeitsbudget-statt-fit-versprechen)
setzt vorgeschlagen höchstens 6 GiB gesamten Prozess-VRAM an. Das bleibt eine
Zielgrenze; kein neuer Komponentenfit, keine FPS-Zusage und kein festgelegtes
Parameterbudget folgen aus der aktuellen Diskussion.

| Anteil | Budgetinterpretation |
| --- | --- |
| Zeitgestempelte Pose/Form/Ausdrucksparameter, Identitätsreferenzen, Quellenbezug | Dauerhaft in CPU-RAM/SSD; nur aktuell benötigte Ausschnitte als begrenzter GPU-Arbeitssatz. Ein wachsender Graph muss nicht vollständig auf der GPU liegen. |
| Gemeinsamer Bildleser und Agentenkern | Modellgewichte plus aktivierte Multiskalenmerkmale, Schleifen und aktuelle Zustände zählen gemeinsam. Ein kleines Parametermodell garantiert keinen kleinen Aktivierungsspeicher. |
| Gesichts-/Hand-/Fußdetails | Hochauflösende kleine Ausschnitte und begrenzte Personenzahl/Arbeitsmenge; geteilter Leser als Designziel. Separate große Modelle nur nach eigenem Speicher-/Latenzprofil. |
| Große Referenzmodelle | [Human3R](https://fanegg.github.io/Human3R/) nennt allein 8 GB Inferenzbedarf. Das passt in dieser berichteten Konfiguration nicht in unser 6-GiB-Prozessziel plus übrigen Agenten. Kein pauschaler Fit der anderen unvermessenen Modelle. |
| Lernen | Referenzen aufnehmen benötigt keinen Backward-Pass. Filtertraining braucht zusätzlich Gradienten, Optimizer und gespeicherte Aktivierungen; eine eigene Budgetprüfung. |

Illustrative Rechnung, keine ausgewählte Repräsentation: 1000 FP32-Koeffizienten
benötigen 4000 Bytes je Person/Zeitpunkt. Bei 10 Hz entstehen 144 MB pro Stunde
und Person, ohne Zeitstempel, Indizes, Quellenbilder und Datenbank-Overhead.
512 FP32-Deskriptorwerte × 20 Referenzen × 1000 Personen ergeben 39,06 MiB
reine Vektoren. Wachsende Historie braucht daher Aufbewahrungs-/Verdichtungsregeln,
aber keinen ebenso wachsenden GPU-Arbeitssatz. Verdichtung ist verlustbehaftet;
Ereignisse/Quellreferenzen und ihr jeweiliger Erhaltungsumfang bleiben explizit.

Gewichtsspeicher und vollständiger Lauf sind verschieden: 100M FP16-Parameter
sind 0,186 GiB reine Gewichte. Mit FP32-Parametern, FP32-Gradienten und zwei
FP32-AdamW-Momenten sind 100M trainierbare Parameter bereits 1,49 GiB persistenter
Zustand, noch ohne Aktivierungen/Workspaces. Andere Precision-/Optimizer-Verträge
haben andere Kosten. Als Aktivierungsbeispiel belegt eine 1920×1080×64-FP16-
Merkmalskarte allein 253,1 MiB; acht solche Frames knapp 2 GiB vor weiteren
Skalen, Lesern und Backward. Das ist kein Vorschlag, diese Karte so anzulegen.

Vorschlag für die Geschwindigkeitspriorität: Bildmerkmale teilen, aktuelle Tracks
günstig fortschreiben, stabile Form selten aktualisieren, Detailausschnitte gezielt
lesen und Graphhistorie außerhalb der GPU halten. Die kürzeren Aufrufintervalle
müssen pro Signal gewählt werden; langsame Formaktualisierung darf schnelle
Lippen-/Lidereignisse nicht entfernen. Nacheinander aufgerufene Modelle behalten
bei GPU-Residenz ihre Gewichte; erst Entladen/CPU-Offload senkt deren gleichzeitigen
Gewichtsbedarf und erzeugt Transfer-/Ladelatenz. Keine kostenlose Echtzeitlösung.

Kleinste noch ausstehende Prüfung: eine konkret festgelegte vollständige Pipeline,
Bildgröße und maximale Personenanzahl mit echten Crops unter dem 6-GiB-Ziel
profilieren. Warm-up, lange Streamingphase, Detailauslösung und gegebenenfalls
vollständige Optimizer-Schritte erfassen; Prozess-VRAM, allokierter/reservierter
Speicher, CPU-RAM und p50/p95-End-to-End-Latenz getrennt berichten. Konkrete
Latenz-/Qualitätsgates vor Laufbeginn setzen. Alte kleine Modelltests validieren
diese neue Personenpipeline nicht. Keine Gewichtsdownloads oder Modelltests in
dieser Budgetklärung; Design, Durchsatz und Erkennungsqualität bleiben offen.

## Räumliches Wissen und parametrische Personen (Diskussion 26./27. September)

Alex fragt, ob räumliches Wissen über Szenen und Entitäten latent oder explizit
(Meshes, Punktwolken, Gaussian Splats) gelernt und abgelegt werden soll, und schlägt
für Personen Skelett-/Poseparameter, Formparameter (SMPL oder neuer) und ein
Gesichtsmodell (FLAME oder neuer) vor. Dies ist ein **Vorschlag**, keine Übernahme,
Implementierung oder Validierung.

**Vorgeschlagene Aufteilung (hybrid):**

| Ebene | Inhalt | Rolle |
| --- | --- | --- |
| Evidenz | Frames, Kamera-/Aktionsprotokolle, ggf. Tiefe | maßgeblich; vorhandener Store |
| Instanzzustand | Pose, Ausdehnung, Stütz-/Enthaltenseinsbeziehungen mit Unsicherheit; bei Personen Form β (langsam, zusammengeführt), Pose θ(t), Ausdruck ψ(t), Platzierung, Herkunft | explizit, versioniert, korrigierbar ohne Gewichtstraining |
| Geometrie-Cache | objektzentrierte Gaussians mit latenten Merkmalen; Mesh aus Parametern bei Bedarf | abgeleitet, neu aufbaubar; Rendern zur Prüfung |
| Latenter Kern | Entitätstokens (Code + Pose), gezielt nachgeladene Gelenk-/Detailtokens | Konzepte bleiben latent, Instanzen explizit |

Begründung: Rendern der Instanzhypothese ist eine unabhängige Prüfung (Vorschlagen
und Prüfen getrennt); Laufzeitkorrektur einer Personenform ist ein kontrollierter Fall
des Gesamtziels (Wissen ohne Nachtraining erwerben und korrigieren). Rein latente
Szenenzustände sind schwer prüfbar; Meshes eignen sich schlecht für Unscharfes;
Punktwolken tragen kein Aussehen; Splats sind speicherintensiv und dynamisch schwierig.
Eine latente, per Strahl-Decoder renderbare Szenenmenge (SRT/OSRT-artig) bleibt
Vergleichsoption.

**Nutzung durch den Agenten (Vorschlag):** (1) Entitätstokens aus β, θ, ψ und Pose für
den Kern; (2) exakte Werkzeuge aus Vorwärtskinematik: Hand-/Kopfposition, Zeige- und
Blickstrahl gegen den Szenengraph; (3) zeitliches Fenster aus der Historie für
Bewegung und Ereignisse. Anwendungen: Zeigen/Blick auf Referenzobjekte abbilden,
Posevorhersage mit Prüfung am nächsten Frame, Gesten als aus wenigen Beispielen
erschlossene Konzepte (dasselbe Induktionsproblem wie §23 des Integrationsplans),
Sprecherzuordnung mit Audio, Form als schwacher Wiedererkennungshinweis, Übergaben
und Abstand beim eigenen Handeln. Vorgeschlagener erster Fall: **Zeigen**.

**Format:** SMPL-X als gemeinsames Körper-/Hand-/Gesichtsformat; SKEL optional für
biomechanische Gelenke; FLAME/SMIRK für detailliertere Gesichter. Zuerst ein externer
Schätzer als Beobachtungsadapter/Lehrer (Pseudolabels) mit Nachoptimierung gegen
Gelenkpunkte/Silhouette; später eigene Köpfe am Multiskalen-Encoder per Destillation.
Kleidung/Haare später optional als Gaussians auf der Körperoberfläche.

**Offene Entscheidungen für Alex:** Forschungslizenzen (SMPL/SMPL-X/FLAME/SKEL nur
nichtkommerziell, Download nach eigener Registrierung); Wahl des Schätzers (zwei
aktuelle Kandidaten mit Größe/Lizenz recherchieren, kein Download ohne Zustimmung);
Priorität gegenüber der Regelinduktion; wichtigste Anwendung. Kleinster erster Test
nach Zustimmung: ein kurzer Webcam-Clip (vorhandenes `experiments/capture_webcam.py`),
Ablage unter der Personen-Entität, Prüfungen: Gelenk-Reprojektion gegen unabhängigen
2D-Detektor, Silhouettenüberlappung, Formstabilität, Revision bei neuen Ansichten,
Speicher/Latenz unter 6 GiB.

**Schätzerrecherche (27.09., 1 Uhr, nur Primärseiten gelesen, nichts heruntergeladen):**

| Kandidat | Ausgabe | Größe / Laufzeit (Autorenangabe) | Lizenz / Voraussetzungen | Einordnung |
| --- | --- | --- | --- | --- |
| [Multi-HMR](https://github.com/naver/multi-hmr) (NAVER, ECCV 2024; Update Feb. 2026) | SMPL-X inkl. Hände/Gesichtsausdruck, mehrere Personen in einem Durchlauf | Checkpoints ViT-S/B/L; 672 px: 29/43/74 ms auf V100 | Code CC BY-NC-SA 4.0; `SMPLX_NEUTRAL.npz` separat nach Registrierung | **Erster Kandidat:** ViT-S/B plausibel unter dem 6-GiB-Ziel, keine eigene Messung |
| [SMPLest-X](https://github.com/MotrixLab/SMPLest-X) (TPAMI 2025) | SMPL-X, stark skaliert | Huge-Gewichte 8,2 GB; YOLOv8x-Detektor zusätzlich | SMPL-X/SMPL-Dateien nötig | Passt in dieser Form nicht ins GPU-Budget; höchstens Offline-Lehrer |
| MediaPipe Pose/Face Landmarker (siehe oben) | 2D/3D-Landmarken, Blendshapes | leicht, CPU-echtzeitfähig | Apache 2.0 | **Unabhängige Prüfreferenz** für Gelenk-Reprojektion, kein SMPL-X |

Vorschlag: Multi-HMR ViT-S als Beobachtungsadapter/Lehrer, MediaPipe als unabhängige
Prüfung der Reprojektion. Beides erst nach Alex' Zustimmung zu Lizenz und Download.
