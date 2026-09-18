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

## Spätere Zuordnung der Techniken

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
