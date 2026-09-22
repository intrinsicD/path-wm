# Latentes Konzeptlernen: Forschungsstand und Voraussetzungen

Quellenrecherche vom 22. September, Abgleich abgeschlossen am 23. September 2026.
Auftrag: den Forschungsstand zunächst allgemein
und abstrakt mit Claude Opus 5.5 bei Aufwand `max` diskutieren. Alex bevorzugt
latente Tokens und interne latente Verarbeitung; er fragt, welche Fähigkeiten
Encoder, Decoder und andere Komponenten vorher benötigen. Diese Präferenz legt
noch keinen konkreten Mechanismus oder Trainingslauf fest.

Die Notiz ordnet öffentlich belegte Methoden und ihre Grenzen. Sie ist keine
vollständige Rangliste aller Benchmarks und kein lokaler Fähigkeitsnachweis.
Zwei abgeschlossene Antworten von Claude und die unabhängige Quellenprüfung
liegen vor; Ausführungsunterbrechungen sind am Ende dokumentiert.

## Was ein Konzept leisten soll

Ein latenter Vektor kann ein Beobachtungsmerkmal, eine einzelne Erinnerung, einen
Auftrag oder eine verallgemeinerbare Regel darstellen. Seine numerische Form legt
diese Bedeutung nicht fest. Für diese Diskussion gilt als **Arbeitskriterium**:
Ein erworbenes Konzept muss neue Fälle unter relevanten Variationen richtig
behandeln, Gegenbeispiele berücksichtigen und mindestens eine weitere passende
Vorhersage oder Entscheidung ermöglichen. Der genaue Umfang dieser Forderung
muss zur untersuchten Konzeptfamilie passen.

Vier Leistungen sollten getrennt geprüft werden:

- **Instanzbindung:** dieselbe Entität über mehrere Beobachtungen verfolgen.
- **Kategorisierung:** mehrere unterschiedliche Instanzen nach einem Kriterium
  zusammenfassen.
- **Relation oder Funktion:** eine Beziehung beziehungsweise Wirkungsregel über
  wechselnde Beteiligte anwenden.
- **Komposition:** bekannte und neu erschlossene Teile für eine weitere Aufgabe
  miteinander verbinden.

Keiner dieser Vorgänge verlangt grundsätzlich eine ausgeschriebene Definition.
Eine mögliche Repräsentation besteht aus latenten Beispielen, einem oder mehreren
Prototypen, relationalen Bindungen oder einem erschlossenen Funktionscode. Auch
mehrere Formen können gemeinsam nützlich sein. Ein einzelner Mittelwert reicht
nicht für jede mehrdeutige, relationale oder kontextabhängige Kategorie.

## Welche Methoden tatsächlich welche Teilfrage beantworten

| Methodenfamilie | Beleg und Aussage | Grenze für das Integrationsziel |
| --- | --- | --- |
| Selbstüberwachtes Merkmalslernen | [DINOv3, August 2025](https://arxiv.org/abs/2508.10104) liefert wiederverwendbare dichte Bildmerkmale; [V-JEPA 2.1, März 2026](https://arxiv.org/abs/2603.14482) lernt räumlich und zeitlich strukturierte Merkmale durch latente Vorhersage. | Gute Merkmale belegen weder selbstständige neue Konzeptbildung noch dauerhafte Wissenskorrektur. Die große Vortrainingsleistung muss bei Wiederverwendung mitgedacht werden. |
| Objektbindung | [DINOSAUR, ICLR 2023](https://dinosaur-paper.github.io/) lernt Objektgruppierung durch Rekonstruktion vortrainierter Merkmale. [Object Concepts Emerge from Motion, September 2026](https://arxiv.org/abs/2609.04348) nutzt Bewegung zur Erzeugung von Instanz-Lernsignalen. | Objektgruppierung ist eine begrenzte Form von Strukturlernen. Die neue Bewegungsarbeit verwendet unter anderem vorhandene Flussschätzer und große Datenmengen; sie belegt keine beliebige Begriffsbildung. |
| Neue Kategorien aus Beispielen | [Prototypical Networks, 2017](https://arxiv.org/abs/1703.05175) lernen zunächst einen Merkmals-/Vergleichsraum und bilden später aus Beispielen Prototypen neuer Klassen. | Starker, einfacher Bezugspunkt für Lernen über Zustand. Der gelernte Raum und das Distanzmodell begrenzen die untersuchten Kategorien. |
| Lernen von Regeln aus Kontext | [Meta-learning for compositionality, Nature 2023](https://www.nature.com/articles/s41586-023-06668-3) trainiert auf wechselnden Aufgaben; im Test werden neue Beispielzuordnungen mit eingefrorenen Gewichten komponiert. [Latent Concept Disentanglement, 2025](https://arxiv.org/abs/2506.16975) untersucht erschlossene Zwischenbegriffe und numerische Funktionsparameter. | Kontrollierte Aufgabenfamilien; teilweise wird bereits vorhandenes Faktenwissen erschlossen. Die numerischen Experimente umfassen kleine, von Grund auf trainierte Modelle, aber keine offene multimodale Begriffswelt. |
| Latente Regeln suchen | [Searching Latent Program Spaces, NeurIPS 2025](https://arxiv.org/abs/2411.08706) lernt implizite Programmcodes und sucht zur Testzeit nach einem Code, der Eingabe-/Ausgabebeispiele erklärt. | Ein direkt passender Mechanismus für latente Regelinduktion; geprüft auf begrenzten Aufgaben zum Lernen aus Beispielen. Die Übertragung auf multimodale Wahrnehmung, dauerhaftes Gedächtnis und Korrektur bleibt offen. |
| Neue relationale Aufgaben | [RDB-PFN, März 2026](https://arxiv.org/abs/2603.03805) trainiert auf synthetischen Aufgaben und nutzt Kontextbeispiele für neue relationale Datenbanken. | Relevant für das Prinzip eines vortrainierten Lernverfahrens; strukturierte Vorhersageaufgaben sind kein Nachweis autonomer Konzeptentdeckung aus natürlichen Sinnesdaten. |
| Latentes Gedächtnislesen | [LatentMem, Februar 2026](https://arxiv.org/abs/2602.03036) erzeugt aus abgerufenen Erfahrungen gelernte Gedächtnistokens. [MemGen, September 2025](https://arxiv.org/abs/2509.24704) trainiert Auslösung und Erzeugung latenter Gedächtnisinhalte für einen eingefrorenen Sprachmodellkern. | LatentMem behält rohe Trajektorien als Erfahrungsbank. MemGen verwendet auch parametrisches Wissen des Erzeugers. Ein eingefrorener Kern beweist nicht, dass das Gesamtsystem neue Fakten dauerhaft ohne Gewichtsänderung erwirbt. |
| Kontinuierliche Denkschritte | [Coconut, Dezember 2024](https://arxiv.org/abs/2412.06769) führt versteckte Zustände wieder als Eingaben zu und trainiert diesen Pfad mit einem Curriculum. | Ergebnisse stammen aus begrenzten Sprachmodellaufgaben und belegen keine allgemeine multimodale Konzeptinduktion. Kausale Eingriffe und Generalisierung müssen zusätzlich geprüft werden. |

Die jüngere [Analyse von Coconut-Varianten, Dezember 2025](https://arxiv.org/abs/2512.21711)
berichtet Abkürzungen über Datensatzmuster und geringe Wirkung bestimmter
Token-Eingriffe. Die Autoren benennen zugleich Grenzen ihrer kausalen Analyse.
Das begründet gezielte Kontrollen, kein pauschales Urteil über latentes Denken.

Ein [Gedächtnis-Pilot vom März 2026](https://arxiv.org/abs/2603.16413) untersucht
trainierte Adapter und laufende Gedächtnisakkumulation bei eingefrorenem Flan-T5-XL.
Ein Datensatz, ein Backbone und deutliche Kapazitätseffekte begrenzen die Aussage.
Der Pilot ist ein Machbarkeitsbeleg für eine Teilfunktion, kein allgemeiner
Fähigkeitsstandard.

## Müssen Encoder und Decoder schon fertig sein?

**Funktionale Voraussetzung und zeitliche Trainingsreihenfolge sind verschieden.**
Wenn der fertige Agent zur Laufzeit neue Konzepte nutzen soll, müssen seine
Wahrnehmung, sein Lern-/Vergleichsverfahren und seine Leser dafür geeignet sein.
Diese Fähigkeiten können während des initialen Trainings gemeinsam entstehen.
Ein bereits separat fertig trainierter Encoder ist keine allgemeine theoretische
Voraussetzung für den Beginn dieses Trainings.

Der Encoder muss die für die jeweilige Begriffsbildung nötigen Unterschiede
zugänglich machen. Ein späterer Leser kann verlorene Unterschiede aus dem
gespeicherten Code allein nicht zuverlässig zurückholen. Deshalb sind gewünschte
Invarianzen und benötigte Details gemeinsam zu prüfen: Was für eine Aufgabe
unwichtig ist, kann für eine spätere Kategorie entscheidend sein.

Ein vollständiger Bild-, Video- oder Audiodecoder ist **keine universelle
Voraussetzung**. Latente Vorhersage kann latente Ziele verwenden; eine
Kategorisierungsaufgabe kann zunächst eine Zuordnung auslesen. Ein Feature-Leser,
Vergleichskopf oder Handlungsleser erfüllt andere Aufgaben als ein Generator
vollständiger Sinnesdaten. DINOSAUR rekonstruiert beispielsweise Merkmale.
Für hochwertige modale Ausgabe wird später ein passender Generator benötigt.

Ein besonders direkter neuer Beleg ist
[LeWorldModel, März 2026, überarbeitet Juni 2026](https://arxiv.org/abs/2603.19312):
Encoder und latenter Prädiktor werden gemeinsam aus Pixeln gelernt, mit
Vorhersageverlust und Regularisierung gegen Kollaps. Das stützt einen kompakten
gemeinsamen Einstieg in den untersuchten Kontrollumgebungen. Es belegt weder
allgemeines Konzeptlernen noch eine bestimmte Eignung für unser Modell.

Gleichzeitig benötigt das Training ein aussagekräftiges Lernsignal: etwa
Vorhersage, mehrere Ansichten, Beispiel-/Gegenbeispielaufgaben, Beziehungen oder
beobachtete Handlungsfolgen. Nur latente Tokens bereitzustellen spezifiziert
weder das gewünschte Verhalten noch einen Schutz gegen Informationsverlust oder
Repräsentationskollaps.

| Trainingsweg | Nutzen | Offene Schwierigkeit |
| --- | --- | --- |
| Gemeinsam von Grund auf | Merkmale und Verbraucher können sich aufeinander abstimmen. [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) belegt gemeinsames Lernen von Weltmodell und Verhalten in seinen Kontrollaufgaben. | Daten, Lernsignale und Stabilisierung; ein scheiterndes Gesamtsystem ist schwer zu diagnostizieren. Dreamer ist kein Nachweis eines einzigen eingefrorenen Agenten für alle Domänen. |
| Gestufter Aufbau | Ein zunächst brauchbarer Wahrnehmungs-/Lesepfad vereinfacht die Untersuchung von Konzept- und Gedächtnisoperationen. | Zu frühes Einfrieren kann fehlende Merkmale festschreiben. Ein späteres gemeinsames Nachjustieren bleibt eine Vergleichsoption. |
| Vortrainierte Merkmale wiederverwenden | Natürliche Eingaben werden ohne eigenes vollständiges Vortraining untersuchbar. | Fremde Vortrainingskosten und Vorannahmen bleiben relevant; latente Schnittstellen müssen tatsächlich kompatibel gemacht werden. |

Für ein begrenztes Forschungsbudget ist der gestufte Vergleich mit brauchbaren
Merkmalen eine **Empfehlung**, keine beschlossene Architektur. Gleiche Vektorbreite
macht zwei Repräsentationen noch nicht gegenseitig lesbar. Gemeinsame Lernziele,
gelernte Übersetzer oder kompatible Leser können diese Verbindung herstellen.

## Wie Lernen ohne laufendes Nachtraining möglich wird

Das initiale Training kann ein Verfahren zum Erschließen neuer Zusammenhänge
lernen. Zur Laufzeit bekommt dieses Verfahren neue Beispiele und verändert
Gedächtnis und Arbeitszustand. Ein vortrainierter Vergleichsraum ermöglicht
beispielsweise neue Prototypen; ein auf wechselnden Aufgaben trainierter Kern
kann Regeln aus einem neuen Beispielkontext erschließen.

Die Gewichte speichern dann unter anderem **Lernoperationen und Vorwissen**;
der veränderliche Zustand trägt **aktuelle Beispiele, Zuordnungen und erschlossene
Hypothesen**. Neuartige Kombinationen und Abstraktionen sind innerhalb der
verfügbaren Merkmale und Operationen möglich. Beliebige neue Wahrnehmungsfähigkeit
oder unbeschränkte Generalisierung folgt daraus nicht.

[Titans](https://arxiv.org/abs/2501.00663) ist als Vergleich wichtig: Seine schnelle
neuronale Erinnerung verwendet gradientenbasierte Speicherupdates. Auch wenn
langsam gelernte Parameter unverändert bleiben, ist das ein anderer Vertrag als
ein Test, bei dem sämtliche lernbaren Gewichte und verhaltensrelevanten Puffer
eingefroren bleiben und ausschließlich expliziter Zustand geändert wird.

**Optimieren eines Konzeptcodes ist eine weitere, getrennte Möglichkeit.**
Ein Modell kann bei festen Gewichten einen veränderlichen latenten Code suchen,
der neue Beispiele erklärt. Das LPN untersucht sowohl Kandidatensampling als
auch Gradienten auf diesem Code. Solche Gradienten ändern den Code, nicht die
gelernten Modellgewichte. Ein Verbot weiteren Gewichtstrainings schließt diese
Möglichkeit daher nicht automatisch aus; ihre zusätzlichen Kosten und ihre
Generalisierung müssen separat geprüft werden.

Für die nächste abstrakte Diskussion ergeben sich drei miteinander kombinierbare
Wege: Beispiele beziehungsweise Prototypen speichern; mit einem vortrainierten
Induktionsverfahren direkt einen Konzeptcode erschließen; oder diesen Code durch
zusätzliche Suche verfeinern. Prototypen sind selbst eine mögliche Form von
Kategorielernen. Ein komplexerer Weg muss seinen zusätzlichen Nutzen gegenüber
ihnen zeigen; ihr Gleichstand widerlegt nicht automatisch den Konzepterwerb.

## Gedächtnis, Korrektur und weitere Rechentiefe

Als **zu prüfender Entwurf** lassen sich mehrere Verantwortungen verbinden:
Beispiele und Belege erhalten; bedarfsgerecht in einen begrenzten Kontext lesen;
latente Hypothesen oder Zusammenfassungen erzeugen; bei neuen Gegenbelegen
betroffene Inhalte revidieren. Komprimierte Ableitungen sollten auf ihre Quellen
zurückführbar bleiben. Exakte Identitäten, Zeitpunkte und Versionen können latente
Inhalte begleiten, ohne jede Bedeutung in natürliche Sprache umzuwandeln.

Dabei sind eine falsche frühere Zuordnung und eine tatsächliche spätere Änderung
der Welt verschiedene Fälle. Im zweiten Fall kann die alte Beobachtung weiterhin
historisch richtig sein. Wiederholte ähnliche Beobachtungen sind außerdem nicht
automatisch unabhängige Belege oder kalibrierte Sicherheit.

Ein fester Encoder vermindert Koordinatendrift. Er verhindert nicht jeden Fehler
durch wechselnden Kontext, veraltete Zuordnungen oder verlustbehaftete Verdichtung.
Beim Wechsel des Encoders müssen alte Erinnerungen weiterhin lesbar bleiben oder
kontrolliert neu codiert beziehungsweise übersetzt werden.

Wiederholte Verarbeitung mit geteilten Gewichten kann mehr Rechentiefe bereitstellen.
Sie benötigt Aufgaben und Lernsignale, bei denen weitere Schritte tatsächlich
helfen. Diffusion ist eine mögliche Familie iterativer Kandidatenverfeinerung.
Aus den hier geprüften Quellen folgt keine bevorzugte Rolle für allgemeines
Konzeptlernen oder Wissenskorrektur; diese Auswahl bleibt offen.

Die stehenden Gestaltungsprinzipien passen hierzu: Quellen einmal vorbereiten,
mehrere Leser zulassen, feine Evidenz neben kompaktem Kontext erhalten, Besitz und
Lebensdauer von Zustand klären und Vorschläge unabhängig prüfen. Erwartete Vorteile
sind Wiederverwendung und überprüfbare Korrektur; Kosten entstehen durch erhaltene
Quellen, Abruf und erneute Verarbeitung. Eine latente Darstellung allein garantiert
weder Kompaktheit noch Geschwindigkeit oder Genauigkeit.

## Welche Vergleiche die offenen Fragen klären würden

1. **Informationszugang:** Kann ein geeigneter Leser die benötigte Unterscheidung
   aus den Merkmalen treffen, bevor wir Fehler der Konzeptinduktion zuschreiben?
2. **Erwerb und Anwendung:** Helfen neue Beispiele bei zurückgehaltenen Instanzen,
   Variationen und passenden Folgeaufgaben? Vergleich mit einfacher Ähnlichkeit,
   Prototypen und direktem Abruf; neue Namen allein reichen nicht.
3. **Nutzung des Konzepts:** Verändern fehlende, vertauschte oder widersprüchliche
   Konzeptinhalte die Entscheidung passend? Ein auslesbarer Code allein zeigt
   keine kausal wirksame Nutzung durch den Agenten.
4. **Erhalt und Korrektur:** Bleibt das erworbene Wissen nach Ablenkung verfügbar?
   Korrigiert ein Gegenbeleg die betroffene Regel und ihre Folgerungen, während
   weiterhin gültiges Wissen erhalten bleibt?
5. **Lern- und Ressourcenvertrag:** Welche Gewichte beziehungsweise Zustände ändern
   sich? Gelten Gewinne auch bei vergleichbarem Informationszugang, Speicher und
   Rechenaufwand? Generalisierung innerhalb einer Aufgabenfamilie und Transfer zu
   einer anderen Familie werden getrennt ausgewiesen.

Diese Prüfungen sind Diskussionsvorschläge. Domäne, Daten, Erfolgsschwellen und
Trainingsbudget sind weiterhin ungewählt. Der nächste gemeinsame Schritt wäre,
die erste zu lernende Konzeptfähigkeit und ihre nötigen Operationen festzulegen.

## Nächste Besprechung: fünf latente Operationen

23. September 2026. Alex möchte die vorgeschlagene Besprechung fortsetzen.
Die folgende Funktionsbeschreibung erläutert die bereits geprüften Methoden;
sie legt weder fünf getrennte Netze noch eine neue Tokenaufteilung fest.
Die konkrete Umsetzung und Lernreihenfolge bleiben Vorschläge.

Der gemeinsame Ablauf wäre: relevante Evidenz lesen, Gemeinsamkeiten und
Unterschiede erschließen, Beteiligte und Rollen zuordnen, eine vorläufige
Verallgemeinerung bilden, sie auf weitere Fälle anwenden und anhand neuer
Evidenz korrigieren. Mehrere dieser Schritte können durch denselben trainierten
Kern und wiederholte Verarbeitung erfolgen. Eine vorgeschriebene sprachliche
Zwischenbeschreibung ist dafür nicht Teil des Entwurfs.

| Operation | Gewünschte Wirkung im latenten Raum | Woran wir ihre Nützlichkeit erkennen würden |
| --- | --- | --- |
| Vergleichen | Unterschiede, Gemeinsamkeiten und Beziehungen zwischen ausgewählten Repräsentationen abhängig von Aufgabe und Kontext erschließen. Ein Ähnlichkeitswert kann genügen; bei komplexeren Fällen kann ein gelernter Leser differenziertere Vergleichsinformation liefern. | Relevante Unterschiede beeinflussen die Entscheidung; irrelevante Variationen innerhalb des untersuchten Umfangs nicht. |
| Binden | Eigenschaften, Beobachtungen und Rollen den richtigen Beteiligten, Quellen und Zeitpunkten zuordnen. | Das Vertauschen von Beteiligten oder Rollen verändert das Ergebnis passend. Gleiche Eigenschaften führen nicht automatisch zur Verschmelzung verschiedener Instanzen. |
| Verallgemeinern | Aus Beispielen und gegebenenfalls Gegenbeispielen eine auf weitere Fälle anwendbare Kategorie, Relation oder Regel erschließen. Der Zustand kann Beispiele, Prototypen oder einen Funktionscode tragen. | Die Hypothese hilft bei zurückgehaltenen Fällen und passenden neuen Kombinationen. Bloßes Wiedergeben der Lernbeispiele genügt dafür nicht. |
| Anwenden | Die erschlossene Repräsentation für Zuordnung, Vorhersage, Kontextwahl oder Handlung tatsächlich verwenden. | Passende Änderungen am Konzeptinhalt verändern die nachfolgende Wirkung; gute Auslesbarkeit allein reicht nicht als Nachweis der Nutzung. |
| Korrigieren | Neue Evidenz zur Prüfung und Revision der betroffenen Zuordnung beziehungsweise Hypothese nutzen; abhängige Ableitungen bei Bedarf erneuern. | Ein Gegenbeleg verändert die betroffene Vorhersage sinnvoll, während weiterhin gültiges Wissen im erklärten Speicher-/Zeithorizont erhalten bleibt. |

**Der offene Kern ist der Schritt vom Vergleich zur Verallgemeinerung.**
Aus wenigen Beispielen können mehrere Regeln zugleich folgen. Der Agent sollte
deshalb Unsicherheit beziehungsweise alternative Erklärungen tragen können,
statt jede Gemeinsamkeit sofort zum dauerhaften Konzept zu erklären. Welche
Erklärung nützlich ist, hängt von weiteren Beispielen, der Aufgabe und den
überprüfbaren Folgen ab. Diese Forderung ist ein Entwurf für Verhalten, noch
keine Entscheidung für eine bestimmte Verteilung oder Zahl von Hypothesen.

Ein möglicher Lernweg innerhalb des bereits besprochenen Meta-Learning-Ansatzes
übt diese Operationen auf wechselnden Aufgaben: Aus einem Teil der Beispiele
einen latenten Zustand bilden und mit ihm weitere Fälle bearbeiten. Das
Grundtraining verändert die Gewichte der dafür verwendeten Funktionen. Bei der
späteren Aufnahme neuer Inhalte könnten diese Funktionen fest bleiben, während
Beispiele, Bindungen und erschlossene Codes wechseln. Geprüft wird insbesondere
die Wirkung eines Codes; es gibt nicht automatisch einen eindeutig richtigen
Zielvektor, den wir ihm vorgeben müssten.

Das Speichern ist eine zusätzliche Verantwortung: Eine gerade gebildete
Hypothese lebt zunächst im Arbeitszustand. Ob sie länger behalten wird, was ihre
Belege sind und wann sie erneut geprüft werden muss, gehört zur Kontext- und
Gedächtnissteuerung. Eine falsche Ausgabe allein lokalisiert den Fehler nicht:
Auch Wahrnehmung, Zuordnung oder Anwendung können falsch sein. Eine tatsächliche
Änderung der Welt darf zudem historisch richtige Evidenz nicht rückwirkend
falsch machen. Deshalb bleiben Hypothesen, tatsächliche Beobachtungen und
ausgeführte Aktionen unterscheidbar.

Die stehenden Prinzipien bleiben wirksam: Evidenz einmal vorbereiten und für
verschiedene Vergleiche wiederverwenden; nur begrenzten Kontext lesen, bei Bedarf
auf erhaltene Details zurückgreifen; Quellen und Lebensdauern explizit führen;
Hypothesenbildung und Prüfung trennen. Der erwartete Nutzen ist wiederverwendbares
Lernen über Zustand. Offen sind dafür nötige Merkmale, Lernsignale, Abrufqualität
und Gesamtkosten. Der kleinste sinnvolle Vergleich bleibt auf eine erklärte
Konzeptfamilie mit neuen Fällen, Gegenbelegen und einfachen Referenzen begrenzt.

## Quellen- und Reviewbelege

Der tatsächliche Hauptaufruf und der abschließende Abgleich bestätigen
`claude-opus-5-5`, jeweils mit explizitem `--effort max`. Der erste Aufruf durfte
nur öffentliche Websuche und Seitenabruf verwenden; seine Nutzungsmetadaten
enthalten zusätzlich ein Haiku-Hilfsmodell. Im abschließenden Aufruf waren alle
Werkzeuge deaktiviert. Dieser bestätigt ausschließlich Opus 5.5.

Im ersten Text waren mehrere Schlussfolgerungen zu weit gefasst. Nach unseren
Einwänden hat Claude ausdrücklich zurückgenommen beziehungsweise eingegrenzt:

- Ein Gleichstand mit Prototypen widerlegt Konzeptlernen nicht; er begrenzt den
  nachgewiesenen Zusatznutzen eines aufwendigeren Verfahrens.
- Feste Gewichte schließen neue kontextabhängige Zwischenrepräsentationen nicht
  aus. Verlorene Eingabeinformation sowie Rechen- und Generalisierungsgrenzen
  bleiben relevante Beschränkungen.
- Codesuche kann mit oder ohne Gradienten erfolgen; Gradienten auf einem Code
  sind keine Änderung der Modellgewichte.
- Eine isotrope Gesamtverteilung schließt klassenabhängige Struktur nicht aus;
  Rekonstruktion garantiert umgekehrt keine vollständige Konzeptsuffizienz.
- Konsolidierung muss nicht notwendigerweise die langsamen Modellgewichte ändern.
  Quellenreferenzen sind eine gut prüfbare Gestaltungswahl, kein allgemeiner
  Unmöglichkeitsbeweis für andere Korrekturverfahren.
- Erfolge von Test-Time-Training bei bestimmten Aufgaben belegen dessen Nutzen
  unter diesen Bedingungen, keine allgemeine Notwendigkeit für neue Begriffe.

Ein empirisch offener Unterschied bleibt als Prognose: Claude erwartet, dass
Gewichtsanpassung bei weiter entfernten Aufgabenfamilien häufiger helfen dürfte.
Das ist mit gleich budgetierten Vergleichen zu prüfen. Weder diese Prognose noch
unsere Übereinstimmung begründet eine neue Modellentscheidung. Claude hat im
kurzen Abschluss die neu zugeführten Quellen nicht erneut selbst abgerufen;
LeWM und LPN wurden hier unabhängig anhand der Primärquellen geprüft.

Öffentliche Quellenkopien, strukturierte Aussagen/Grenzen und der genaue externe
Auftrag liegen unter `runs/reviews/latent_concepts_sota_20260922/`.
Die erste ausführliche Gegenprüfung endete nach 700 Sekunden ohne Antwort.
Ein enger gefasster Wiederholungsaufruf wurde beim vom Nutzer gemeldeten Logout
unterbrochen. Nach Prüfung der erhaltenen Dateien und nicht mehr vorhandenen
Prozesse wurde nur dieser kurze Abgleich erneut ausgeführt und erfolgreich
abgeschlossen. Fehlende Antworten wurden nicht als Zustimmung gewertet.
Keine privaten Projektdateien oder lokalen Versuchswerte wurden an Claude gesendet.
Es wurde kein Modelltraining oder lokaler Fähigkeitsversuch gestartet.
