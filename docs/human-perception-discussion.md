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
