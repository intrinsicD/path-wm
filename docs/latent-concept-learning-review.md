# Latentes Konzeptlernen: Forschungsstand und Voraussetzungen

Quellenrecherche vom 22. September, Abgleich abgeschlossen am 23. September 2026.
Auftrag: den Forschungsstand zunächst allgemein
und abstrakt mit Claude Opus 5.5 bei Aufwand `max` diskutieren. Alex bevorzugt
latente Tokens und interne latente Verarbeitung; er fragt, welche Fähigkeiten
Encoder, Decoder und andere Komponenten vorher benötigen. Diese Präferenz legt
noch keinen konkreten Mechanismus oder Trainingslauf fest.

Die Notiz ordnet öffentlich belegte Methoden und ihre Grenzen. Sie ist keine
vollständige Rangliste aller Benchmarks und kein lokaler Fähigkeitsnachweis.
Zur ursprünglichen Frage liegen zwei abgeschlossene Antworten von Claude und eine
unabhängige Quellenprüfung vor; Ausführungsunterbrechungen sind am Ende dokumentiert.
Die ergänzende Richtungsdiskussion umfasst zwei weitere abgeschlossene Antworten.
Die spätere [Richtungsprüfung](#richtungsprüfung-geteilte-struktur-über-instanzen)
berücksichtigt Alex’ Präzisierung: „Konzept“ meint zunächst etwas Übergeordnetes
zu Instanzen; Kategorien, Transformationen und Aktionen sind mögliche Unterfälle.

## Was ein Konzept leisten soll

Ein latenter Vektor kann ein Beobachtungsmerkmal, eine einzelne Erinnerung, einen
Auftrag oder eine verallgemeinerbare Regel darstellen. Seine numerische Form legt
diese Bedeutung nicht fest. Für diese Diskussion gilt als **Arbeitskriterium**:
Eine erworbene Abstraktion soll neue Fälle unter den für ihre Nutzung relevanten
Variationen sinnvoll behandeln können. Eine neue Kategorie, mit der weitere
Instanzen richtig zugeordnet werden, ist bereits ein begrenzter Konzeptnachweis.
Mehrere Nutzungsarten, dauerhafte Speicherung und gezielte Revision gehören zum
breiteren Agentenziel; sie sind keine Definitionspflicht jedes einzelnen Codes.
Der genaue Nachweis muss zur untersuchten Abstraktionsfamilie passen.

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
Ebenso außerhalb dieses Vertrags liegen Verfahren, die pro Instanz Gewichte
anpassen (Pivotal Tuning, DreamBooth, LoRA je Identität).

**Optimieren eines Konzeptcodes ist eine weitere, getrennte Möglichkeit.**
Ein Modell kann bei festen Gewichten einen veränderlichen latenten Code suchen,
der neue Beispiele erklärt. Das LPN untersucht sowohl Kandidatensampling als
auch Gradienten auf diesem Code. Solche Gradienten ändern den Code, nicht die
gelernten Modellgewichte. Ein Verbot weiteren Gewichtstrainings schließt diese
Möglichkeit daher nicht automatisch aus; ihre zusätzlichen Kosten und ihre
Generalisierung müssen separat geprüft werden.
Dasselbe gilt für Instanzcodes: DeepSDF-artige Auto-Decoder bestimmen den Code
einer neuen Instanz per Gradientenabstieg bei eingefrorenem Decoder, als Alternative
zur Vorhersage oder Aggregation per Encoder ([Ideensammlung](compact-instance-memory-ideas.md)).
Die vorhandene P3D-Verfeinerung optimiert Pixel gegen den eingefrorenen Encoder und
ist kein Beispiel dafür.

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

## CLIP-Differenzen und latente Transformationen

23. September 2026. Alex fragt, ob die Analogie A → B, C → ? als Differenz
oder als gelernte nichtlineare Transformation im Embeddingraum verstanden werden
kann und ob Konzepte grundsätzlich so gelernt werden müssen. Dies präzisiert die
vorherige Diskussion; es wählt noch keine Architektur aus.

Mit z_X = E(X) ist z_D ≈ z_C + (z_B − z_A) eine übertragene Verschiebung.
Die Subtraktion ist eine lineare Rechenoperation; daraus folgt keine allgemein
additive Semantik. Der Differenzvektor enthält zunächst alle im Encoder sichtbaren
Unterschiede des Paars, auch aufgabenirrelevante. Die SIMAT-Arbeit berichtet
begrenzte Eignung unveränderter CLIP-Embeddings für ihre textdefinierten
Delta-Transformationen und Verbesserungen durch gezieltes Finetuning.
*Finetuning CLIP to Reason about Pairwise Differences* lernt ausdrücklich eine
Ausrichtung von Bilddifferenzen und beschreibenden Texten. Beide Quellen begründen,
warum Analogiestruktur eine zu prüfende beziehungsweise zu lernende Eigenschaft
ist; sie beweisen keine Unmöglichkeit einzelner Analogien in unverändertem CLIP.
Quellen: [SIMAT](https://arxiv.org/abs/2112.03162),
[paarweise Unterschiede](https://arxiv.org/abs/2409.09721).

Allgemeiner kann ein trainierter Kern aus Beispielpaaren und Kontext einen
Relations- beziehungsweise Operationscode r erschließen und ihn auf einen neuen
Ausgangszustand anwenden: r = G(Beispiele, Kontext), z_D = F(z_C, r).
Die Verschiebung F(z,r) = z+r ist ein Spezialfall. Schon lineare beziehungsweise
affine Abbildungen erweitern diese Familie; nichtlineare Funktionen erlauben
zustandsabhängige Änderungen. Die beiden Funktionsrollen erfordern weder separate
Netze noch explizite sprachliche Relationsnamen. G und F können trainierte feste
Gewichte behalten, während r bei neuen Beispielen wechselt. Ein neues r ist nicht
automatisch ein neues separat trainiertes Netz. *Searching Latent Program Spaces*
belegt eine verwandte Inferenz und Suche in latenten Programmcodes in begrenzten
Programmsynthese-/ARC-Aufgaben, keine allgemeine multimodale Konzeptfähigkeit.
Quelle: [LPS](https://arxiv.org/abs/2411.08706).

Nicht jedes Konzept ist eine Transformation: Kategorien können durch Beispiele,
Prototypen oder Zugehörigkeitsfunktionen repräsentiert sein; Beziehungen können
Kompatibilität zwischen Beteiligten beschreiben, ohne einen eindeutigen Zielzustand
zu bestimmen. Ein Operator passt besonders zu Veränderungsregeln. Nichtlinearität
und semantische Qualität sind zudem verschiedene Eigenschaften; nach einer
geeigneten Umkodierung kann eine zunächst komplizierte Transformation einfach sein.
Keine nachgelagerte Funktion kann die vom Encoder verworfene Information aus dem
Embedding allein eindeutig wiederherstellen.

Ein einzelnes Paar bestimmt die gewünschte Regel im Allgemeinen nicht eindeutig.
Mehr Beispiele, Gegenbeispiele, Aufgabenbezug und gelernte Vorannahmen begrenzen die
möglichen Erklärungen. Der vorgeschlagene Lernanreiz ist die Anwendung auf neue
Fälle bei Erhalt der innerhalb der Aufgabe irrelevanten Merkmale. Dafür können
Beispielpaare und Vorhersage-/Kontrastsignale genügen; handbenannte Konzepte oder
vorgegebene richtige Codevektoren sind keine allgemeine Voraussetzung. Bloß einen
leistungsfähigen nichtlinearen Mapper einzusetzen garantiert diesen Transfer nicht.
Aufgaben, Lernsignal, Repräsentation und Budget bleiben offen. Vorbereitete Evidenz
wird wiederverwendet, Kontext bleibt begrenzt und erschlossene Codes bleiben von
beobachteter Evidenz unterscheidbar.

## Bekannte Konzepte als Transfersignal

23. September 2026. Alex schlägt vor, bekannte Konzepte zu verwenden und
Differenzen ausdrücklich so zu lernen, dass ihre Übertragung das richtige
Resultat erzeugt. Das ist ein Vorschlag für ein Lernsignal, noch keine Auswahl
von Daten, Architektur, Loss-Gewichten oder Trainingsbudget.

Eine passende Aufgabe besteht aus gültigen Analogien (A, B, C, D): Die Beziehung
von A nach B soll auch von C nach D gelten. Bekannte Regeln oder kontrollierte
Veränderungen können diese Zuordnung für die Datenerstellung liefern; bloße
Konzeptnamen liefern noch keine vollständigen und eindeutigen Zielpaare. Der
Lernende kann ausschließlich die latenten Beispiele sehen, während die Regel-ID
nur dem Datensatzbau und der Auswertung dient. D ist Ziel und darf nicht in den
Eingabepfad gelangen.

Mit z_X = E(X) kann r = G(z_A, z_B) aus dem Beispielpaar entstehen; ein
Anwender sagt z_D_hat = F(z_C, r) vorher. Ein Transferverlust bewertet die
Übereinstimmung mit einer geeigneten Zielrepräsentation von D. Werden mehrere
Beispiele zur eindeutigen Bestimmung der Beziehung gebraucht, kann G diese als
begrenzten Kontext lesen. Das Modell lernt über die Wirkung des Codes, ohne dass
wir einen richtigen Codevektor oder sprachliche Konzeptbeschreibung vorgeben.

Es gibt zwei unterschiedliche geometrische Festlegungen: Für echte
Vektordifferenzen kann das Training z_B − z_A ≈ z_D − z_C fördern und damit
gezielt eine additive Geometrie lernen. Bei einem nichtlinearen Anwender müssen
die rohen Differenzen nicht gleich sein; derselbe Relationscode muss auf beiden
Ausgangszuständen die richtige Wirkung haben. Die lineare Bedingung ist deshalb
eine zu vergleichende Einschränkung, kein notwendiger Zusatz zu jedem Operator.

Ein nackter Verlust zwischen frei gemeinsam trainierten Embeddings ist ungenügend:
E(X) = konstant erfüllt sowohl die Differenzgleichheit als auch einen passend
konstanten Vorhersageverlust. Eine eingefrorene informative Zielrepräsentation ist
eine mögliche Startbedingung; gemeinsames Lernen benötigt geeignete zusätzliche
Informations-/Aufgabenziele oder andere wirksame Kollapsvermeidung. Allein das
Stoppen eines Zielgradienten ist keine allgemeine Garantie. Ein vollständiger
Modaldecoder ist für eine geeignete latente Zielprüfung nicht zwingend nötig;
ein Encoder muss aber genau die betroffenen Merkmale unterscheiden können.

Der zentrale Transfervergleich trennt neue Inhalte bei bekannten Regeln von
neuen Kombinationen und bisher ungesehenen Regeln. Auch gut passende Ergebnisse
auf bekannten Regel-IDs belegen noch keine Aufnahme neuer Konzepte. Für das
Runtime-Ziel müssten G und F nach dem Grundtraining fest bleiben, während r aus
neuen Beispielen erschlossen und geprüft wird. Passende Kontrollen verwenden
unabhängige Inhalte in Beispiel und Anfrage, mehrere Beziehungen pro Anfrage,
vertauschte beziehungsweise fehlende Codes, zurückgehaltene Kombinationen und
Erhalt der innerhalb der Aufgabe unveränderten Merkmale. Diese Bedingungen
erschweren das Kopieren von B und das Ignorieren des Codes; sie garantieren keine
universelle Generalisierung. Vergleichsverfahren bleiben einfache Übersetzung,
Beispiel-/Prototypennutzung und eine Vorhersage ohne Relationsinformation.

Das Grundprinzip wurde bereits direkt untersucht: *Deep Visual Analogy-Making*
(Reed et al., NeurIPS 2015) trainiert gültige A:B::C:D-Analogien mit Ausgabe D,
vergleicht Vektoraddition mit ausdrucksstärkeren Interaktionen und untersucht
Formen, Sprites und Automodelle. Die dortige Bildrekonstruktion ist ein konkretes
Verfahren, keine Voraussetzung jeder latenten Transferaufgabe.
Quelle: [Originalarbeit](https://papers.neurips.cc/paper_files/paper/2015/file/e07413354875be01a996dc560274708e-Paper.pdf).
Die bereits besprochene [latente Programmsuche](https://arxiv.org/abs/2411.08706)
ist ein verwandtes Beispiel für aufgabenspezifische Codes; keine dieser Arbeiten
validiert unser gesamtes integriertes Gedächtnis-/Korrekturziel.

Stehende Prinzipien: vorbereitete Evidenz innerhalb gültiger Gewichtsstände
wiederverwenden, Quellen und zurückgehaltene Ziele trennen und dieselbe begrenzte
Kontextschnittstelle trainieren, die später verwendet wird. Der erwartete Nutzen
ist direkt trainierte relationale Übertragung; Aufwand und Grenzen liegen unter
anderem in der Qualität der Paarungen, Merkmalsabdeckung und zusätzlichen
Trainingsbeispielen. Als kleinster Vergleich käme eine klar begrenzte Regelfamilie
mit separaten Inhalts-/Regelsplits und gleichen Rechenbudgets infrage; dies ist
noch keine Freigabe für Implementierung oder einen Lauf.

## Konzept, Objektverständnis und latente Aktionen

23. September 2026. Alex fragt nach dem Begriff eines Konzepts, der Darstellung
von Objektverständnis und danach, ob Differenzen zwischen latenten Aktionen ein
geeigneterer Ansatz wären als Bilddifferenzen. Die folgende Arbeitsdefinition ist
ein Vorschlag für unseren Entwurf, keine allgemein abschließende Begriffstheorie.

Ein Konzept ist hier eine wiederverwendbare Abstraktion, die relevante
Gemeinsamkeiten und Unterschiede über einzelne Fälle hinweg erfasst und dadurch
Zuordnung, Erwartungen oder Handlungen auf neue Fälle übertragbar macht. Ein
Konzept kann Kategorie, Eigenschaft, Beziehung oder Regel betreffen. Ein
konkretes Objekt ist zunächst eine Instanz mit Identität und veränderlichem,
teilweise unbekanntem Zustand; ein Objektkonzept liefert verallgemeinerbare
Erwartungen über solche Instanzen. Dafür muss kein einzelner Vektor das gesamte
Wissen enthalten: Seine Bedeutung entsteht auch durch den trainierten Leser,
das Dynamikmodell und abrufbare Evidenz. Diese Funktionsunterscheidung legt weder
separate Module noch manuell benannte symbolische Kategorien fest.

Objektverständnis würden wir an mehreren zusammenhängenden Fähigkeiten prüfen:
Identität über Ansichts-/Zustandsänderungen und Verdeckung hinweg verfolgen,
Eigenschaften und Beziehungen unterscheiden, natürliche Veränderungen und
Handlungsfolgen unter erklärten Bedingungen vorhersagen sowie Unsicherheit und
Gegenbelege berücksichtigen. Keine einzelne Ähnlichkeitsmessung oder Probe
validiert allgemeines Verständnis. Objektbezogene Dynamik ist ein etablierter
Forschungsansatz: C-SWM strukturiert Zustände in Objekte und Beziehungen und lernt
Vorhersagen kontrastiv in begrenzten Umgebungen. PLATO prüft einzelne physikalische
Konzepte mittels Erwartungsverletzungen; seine Segmentierung und Zuordnung über
die Zeit werden durch Ground-Truth-Masken bereitgestellt. Daraus folgt keine
allgemeine Notwendigkeit einer bestimmten Slotarchitektur.
Quellen: [C-SWM](https://arxiv.org/abs/1911.12247),
[PLATO](https://www.nature.com/articles/s41562-022-01394-8).

Latente Aktionen können einen wichtigen Teil dieser Zusammenhänge tragen. Ihre
Differenz allein legt die Bedeutung jedoch ebenso wenig fest wie eine beliebige
Bilddifferenz. Zu unterscheiden sind motorischer Befehl beziehungsweise Bewegung,
beabsichtigter Effekt, beobachteter Effekt und ein aus Übergängen erschlossener
Code. Gleiche Befehle können je nach Zustand andere oder keine Wirkungen haben;
verschiedene Befehle können ein ähnliches Ziel erreichen. Welche Fälle derselbe
Aktionscode zusammenfasst, hängt deshalb von der gewählten Abstraktion ab.

Die allgemeinere Kandidatenbeziehung ist p(z_next | z, u, context): ein Modell
der möglichen Folgezustände bei gegebenem Objekt-/Weltzustand und latenter Aktion.
Beim partiell beobachteten Fall trägt z auch Unsicherheit beziehungsweise Historie.
Kontext enthält relevante andere Beteiligte und Ausführungsbedingungen. Sinnvolle
Vergleiche untersuchen Wirkungen derselben Aktion in verschiedenen Zuständen oder
verschiedener Aktionen im gleichen Zustand. Eine passende Aktionsgeometrie kann
solche Vergleiche vereinfachen, muss aber gelernt beziehungsweise geprüft werden.
Eine nichtlineare Umkodierung verändert Vektordifferenzen, ohne das dargestellte
Verhalten ändern zu müssen; Codebuchnummern lassen sich sogar beliebig umbenennen.
Subtraktion ist daher keine von selbst bedeutungstragende Operation.

Genie zeigt, dass ein aus Videos gelerntes diskretes Aktionscodebuch ein
Dynamikmodell für interaktive Generierung steuern kann. Der dortige latente
Aktionsencoder nutzt auch das Folgeframe im Training; zur Laufzeit wählt der
Nutzer einen Code. Das Verfahren beweist weder semantische Additivität noch die
eindeutige Identifikation tatsächlicher physikalischer Ursachen. Bei Anwendung
auf reale Steuerung wird zusätzlich eine Abbildung zu tatsächlichen Aktionen
benötigt. Quelle: [Genie](https://arxiv.org/html/2402.15391v1).

Eine sinnvolle Fortsetzung des vorgeschlagenen Transfersignals wäre, über
verschiedene Objekte und Kontexte hinweg zutreffende Handlungsfolgen zu lernen und
zu prüfen, welche gemeinsamen latenten Eigenschaften diese Vorhersagen tragen.
Für kausale Aussagen sind entsprechend kontrollierte Eingriffe oder explizite
Identifikationsannahmen nötig: Ein aus beobachteter Veränderung erschlossener Code
kann auch unbeeinflussbare Ereignisse bündeln. Rein visuelle Vorhersage macht ihn
nicht automatisch zu einer ausführbaren Aktion. Beobachtungsdaten bleiben wertvoll;
Verständnis wird hier nicht auf eigene motorische Erfahrung reduziert.

Stehende Prinzipien: Wahrnehmung, Objektzustand und Handlung gemeinsam nutzen,
vorbereitete Evidenz bei gültigem Zustand wiederverwenden, beobachtete und nur
vorgestellte Folgen samt Quellen getrennt halten und Kontext-/Rechenbudgets
begrenzen. Als kleinster Vergleich böten sich zurückgehaltene Objekte/Zustände,
kontrollierte Handlungsvariationen und Vorhersagen ohne beziehungsweise mit
vertauschter Aktionsinformation an; unberührte Beteiligte und Identität müssen im
geprüften Umfang erhalten bleiben. Der erwartete Nutzen sind übertragbare
Verhaltenszusammenhänge; zusätzliche Datenabdeckung, verdeckter Zustand und
Identifikation der Aktionsbedeutung sind offene Kosten beziehungsweise Grenzen.
Keine Datenwahl, Modulaufteilung oder neue Modellfähigkeit ist damit beschlossen.

## Richtungsprüfung: geteilte Struktur über Instanzen

23. September 2026. Alex erläutert, dass „Konzept“ zunächst als Platzhalter für
etwas Übergeordnetes zu einzelnen Instanzen gemeint war. Er bittet um eine
kritische Richtungsdiskussion mit Claude Opus 5.5 bei Aufwand `max`. Diese
Begriffsklärung ist eine Nutzerpräzisierung; die daraus abgeleitete Methodik bleibt
ein Vorschlag. Bisherige Analogie- und Aktionsideen sind damit Unterfälle und
werden weder verworfen noch als allgemeine Lösung übernommen.

**Richtungsurteil:** Das allgemeine Ziel ist das Bilden und Nutzen geteilter,
wiederverwendbarer Struktur. Die vorherige Diskussion wurde stellenweise zu eng,
als sie diese Struktur hauptsächlich als Veränderungsregel oder Handlung behandelte.
Der latente Ansatz bleibt sinnvoll als Forschungspräferenz; seine Nützlichkeit muss
über erklärte Übertragung und Nutzung geprüft werden. Aus „übergeordnet“ folgt
weder ein einzelner Prototyp noch eine feste Hierarchie oder ein universeller
Code, der jede Art von Wissen allein trägt.

| Funktionale Unterscheidung | Gemeinte Rolle, ohne neue Module festzulegen |
| --- | --- |
| Instanz | Ein einzelner Fall beziehungsweise eine gebundene Entität; kein bloßes Bild davon. |
| Zustand und Beobachtung | Was momentan gilt beziehungsweise welche begrenzte Evidenz darüber vorliegt. |
| Geteilte Struktur | Eine über mehrere Fälle nützliche Eigenschaft, Gruppierung, Beziehung, Regel oder ein Muster von Rollen und Wechselwirkungen. |
| Gelerntes Verfahren | Die trainierte Fähigkeit, Evidenz zu lesen, Gemeinsamkeiten zu erschließen, anzuwenden und zu aktualisieren. |
| Veränderlicher Wissenszustand | Die aktuellen Beispiele, Bindungen, Hypothesen, Codes und gegebenenfalls gespeicherten Zusammenfassungen. |

Mehrere Abstraktionen können sich überlappen: dieselben Instanzen können nach
unterschiedlichen Merkmalen oder Nutzungsfragen unterschiedlich zusammengehören.
Die Forschung zur [Kreuzkategorisierung](https://www.sciencedirect.com/science/article/pii/S0010027711000709)
modelliert solche Mehrfacheinteilungen. Das motiviert Offenheit für überlappende
Struktur, nicht die Übernahme der dortigen konkreten Modellierung.

### Methodische Alternativen

- **Beispiele und Prototypen:** Neue Beispiele oder daraus erschlossene
  Zusammenfassungen ermöglichen neue Zuordnungen. Prototypical Networks sind ein
  konkreter Fall mit gelernter Metrik und vorgegeben gruppierten Supportbeispielen;
  sie entdecken nicht selbst beliebige Gruppierungskriterien. Ein stärkerer,
  kontextabhängiger Leser kann mit Beispielen auch andere Beziehungen erschließen.
  Die Grenzen eines Mittelwertklassifikators gelten nicht automatisch für diese
  gesamte Familie. [Primärquelle](https://arxiv.org/abs/1703.05175).
- **Erschlossene latente Codes:** Aus relevanten Beispielen einen Zustand bilden,
  der weitere Abfragen bedingt. Ein solcher Code kann eine Kategorie, eine
  Funktion oder eine Hypothese tragen. Das G/F-Schema ist nicht auf A→B-Paare
  beschränkt. Neural Processes und latente Programmsuche belegen begrenzte
  Varianten, keine allgemeine typenübergreifende Entdeckung oder Persistenz.
  [Neural Processes](https://arxiv.org/abs/1807.01622),
  [LPN](https://arxiv.org/abs/2411.08706).
- **Strukturierte beziehungsweise zusammengesetzte Modelle:** Geteilte Faktoren,
  Rollen, Beziehungen oder Teilprogramme können neue Kombinationen erklären.
  Diese Struktur kann latent oder explizit sein. Keine der beiden Formen hat
  allein aufgrund ihrer Darstellung die beste Kompositionalität. Explizite
  Formen können leichter prüfbar sein; Suche, Bindung und Lernbarkeit bleiben
  jeweils zu vergleichen. [MLC](https://www.nature.com/articles/s41586-023-06668-3)
  belegt erlernte kompositionelle Nutzung in seinen Aufgaben, nicht jede denkbare
  Strukturform oder Aufgabenfamilie.

Diese Alternativen überschneiden sich. Beispiele als Belege und Codes als
Zusammenfassungen sind eine plausible Kombination, noch kein beschlossener Hybrid.
Die Frage nach nützlicher geteilter Struktur kommt vor einer pauschalen Wahl
zwischen Differenzvektor, Operator, Prototyp oder Graph.

### Rolle der Aktionen und des Lernsignals

Wahrnehmung liefert Evidenz über Instanzen und Zustände. Erschlossene Abstraktionen
können diese Zuordnung und weitere Erwartungen beeinflussen; neue Evidenz kann die
Abstraktion wiederum ändern. Diese Abhängigkeiten müssen nicht als starre
Trainingsreihenfolge oder als getrennte Netze umgesetzt werden. Ein abgeschlossener
vollständiger Modaldecoder bleibt keine allgemeine Vorbedingung.

Aktionen nutzen manche Abstraktionen, erzeugen zusätzliche Beobachtungen und können
bestimmte Hypothesen unterscheiden. Handlungsfolgen sind damit ein wichtiges,
aber auf passende Fälle begrenztes Lernsignal. Kategorien und beobachtbare Muster
können auch ohne eigene Intervention gelernt und geprüft werden. Aussagen über
physikalische Eingriffswirkungen benötigen passende Daten oder explizite
Identifikationsannahmen; eine brauchbare Gruppierung ist noch kein kausaler Beleg.

Alex' Vorschlag, bekannte Konzepte für überprüfbaren Transfer zu verwenden, bleibt
nützlich. Allgemeiner könnte ein Lernfall Beispiele einer geteilten Struktur
liefern und anschließend ihre Nutzung an weiteren Fällen verlangen. Dabei ist
zu erklären, ob die Gruppierung beziehungsweise Regel vorgegeben ist oder das
Modell das relevante Kriterium selbst erschließen muss. Ähnliche Ergebnisse
können sonst sehr verschiedene Fähigkeiten verdecken. Zusätzliche Abfragearten
sind hilfreich für das breite Ziel, verhindern aber allein weder kollabierte
latente Ziele noch das Ignorieren des erschlossenen Codes.

### Was als neuer Wissensgewinn gelten kann

Feste Gewichte können Verfahren bereitstellen, die aus neuen Beispielen neue
Zwischenrepräsentationen und zusammengesetzte Hypothesen berechnen. Information,
Kapazität, Rechenbudget und gelernte Vorannahmen begrenzen dies; eine neue
Aufgabenfamilie erzwingt nicht logisch eine Gewichtsänderung. Wie weit ein
trainiertes Verfahren tatsächlich trägt, bleibt eine empirische Frage.

Zu trennen sind neue Instanzen bekannter Muster, neue Kombinationen bekannter
Bestandteile, zurückgehaltene Muster innerhalb geübter Familien und weiter entfernte
Aufgabenfamilien. Diese Grenzen müssen für einen Vergleich konkret beschrieben
werden; es gibt keinen aus Vektorabständen automatisch folgenden Neuheitstest.
Abruf von Beispielen und anschließende Verallgemeinerung können zusammen eine neue
nützliche Abstraktion realisieren. Gleichstand mit einer einfachen Referenz begrenzt
den Zusatznutzen, widerlegt aber nicht die begrenzte Lernfähigkeit selbst.

Der vorgeschlagene gemeinsame Nachweis ist: Neue Evidenz verändert einen nutzbaren
Wissenszustand; dieser verbessert passende zurückgehaltene Fälle, bleibt im
vereinbarten Zeit-/Speicherumfang verfügbar und kann bei Gegenevidenz gezielt
korrigiert werden. Variierte Beispiele bei gleicher Abfrage sowie Entfernen,
Vertauschen und Wiederherstellen der betroffenen Zustände prüfen deren tatsächliche
Nutzung. Dauerhafte, autonome und über mehrere Abstraktionsarten nutzbare
Wissensbildung ist weiterhin Ziel und kein durch diese Literatur etablierter
Gesamterfolg. Auch eine wissenschaftliche Neuheitsbehauptung wird nicht erhoben.

Die nächste Entwurfsfrage lautet deshalb: **Welches Wissen soll von einzelnen
Fällen auf andere Fälle übertragbar werden, und welche Unterschiede muss die
jeweilige Verallgemeinerung dabei beachten?** Einige unterschiedliche Familien
könnten eine gemeinsame funktionale Schnittstelle prüfen; eine bestimmte Anzahl,
Repräsentation oder Datenwahl ist damit nicht vorgegeben.

Stehende Prinzipien: Quellen/Evidenz wiederverwenden, begrenzten Kontext lesen,
Instanzdetails bei Bedarf erhalten und abrufen, vorläufige Abstraktionen mit ihren
Belegen und Lebensdauern führen und ihre Nutzung separat prüfen. Dies verspricht
Wiederverwendung statt isolierter Einzelfalllösungen; Speicher-, Such- und
Aktualisierungskosten sowie verlustbehaftete Zusammenfassungen sind zu begrenzen.
Der kleinste Vergleich sollte eine erklärte Übertragungsfrage mit einfachen
Referenzen und identischen Ressourcen prüfen, bevor weitere Mechanismen folgen.

### Belege der erneuten Claude-Diskussion

Review und gezielter Abgleich sind abgeschlossen. Beide Ausführungsbelege bestätigen
`claude-opus-5-5`, explizit mit `--effort max`, ohne zusätzliche Modelle oder Tools.
Verwendet wurden ausschließlich abstrahierte methodische Fragen und öffentliche
Quellen; die Primärquellenprüfung erfolgte hier unabhängig. Rohtexte und
Ausführungsbelege liegen unter `runs/reviews/concept_abstraction_direction_20260923/`.
Der [dauerhafte Prüfbeleg](../ara/evidence/tables/concept_abstraction_review_2026-09-23.json)
bindet Quellen, Text-Hashes, Ausführungen und verbleibende Unsicherheiten.

Claude hat im Abgleich fünf Einwände angenommen beziehungsweise eingegrenzt:

1. Grenzen einer festen Metrik oder eines Mittelwertprototyps gelten nicht für jeden
   gelernten Beispielsleser. Explizite Komposition garantiert darstellbare Kombinationen
   ihrer Grammatik, aber weder korrekte Inferenz noch bessere Generalisierung.
2. Auch feste Gewichte können höherstufige Hypothesen im veränderlichen Zustand
   erschließen. Vorteile von Test-Time-Training bei weiter entfernten Aufgaben
   bleiben eine überprüfbare Prognose, keine allgemeine Notwendigkeit.
3. Mehrfachnutzung ist ein sinnvoller breiter Nachweis, keine Definition jedes
   Konzepts und für sich weder Kollaps- noch Memorierungsvermeidung.
4. Abruf und Verallgemeinerung bilden keine Gegensätze; Gleichstand mit einer
   einfachen Referenz widerlegt begrenztes Konzeptlernen nicht.
5. Beobachtende Kategorien und Vorhersagen können unter erklärten Annahmen nützlich
   gelernt werden. Besondere Identifikationsanforderungen betreffen insbesondere
   kausale Eingriffseffekte; eigene Aktionen definieren nicht sämtliche Konzepte.

Es bleibt kein prinzipieller Dissens über die hier vorgeschlagene Richtung. Offen
sind insbesondere die Reichweite über Aufgabenfamilien hinweg und die Eignung eines
Zustandsformats für mehrere Arten von Abstraktion. Claude bevorzugt zusätzlich einen
explizit strukturierten Vergleichsarm wegen seiner leichteren Prüfbarkeit. Das ist
hier ein optionaler methodischer Vergleich, kein Fähigkeitsvorrang und keine Abkehr
von Alex’ Präferenz für latente Verarbeitung. Keine Architektur und kein Lauf ist
beschlossen; Übereinstimmung ersetzt keinen empirischen Nachweis.

## Prüfvorschlag: gemeinsame Repräsentation über Abstraktionsarten

23. September 2026. Alex fragt, wie die offene Eignung einer gemeinsamen
Repräsentationsform konkret geprüft werden kann. Folgendes ist ein Vorschlag,
keine gewählte Architektur, kein gestarteter Lauf und kein Ergebnis.

**Hypothese:** Ein gemeinsam gelerntes Zustandsformat mit gemeinsamem Verfahren
zum Erschließen, Anwenden und Aktualisieren kann mehrere Abstraktionsfamilien an
neuen Fällen mit vertretbarem Qualitäts- und Ressourcenverlust gegenüber passenden
Spezialisten tragen. Gleiche Vektorlänge allein belegt weder gemeinsame Semantik
noch eine gemeinsame nutzbare Verarbeitung. Getrennte Codes für unterschiedliche
Inhalte sind erlaubt; ein einziger Code für sämtliches Wissen ist nicht gefordert.

**Kleinste Umgebung:** Eine kontrolliert erzeugte Objektwelt mit Farbe, Form,
Größe, Position, Behältern und Schlüsseln. Dieselben Instanzen tragen überlappende
Abstraktionen: Kategorien, geordnete Beziehungen und bedingte Zustandsänderungen.
Eine spätere Erweiterung prüft zusammengesetzte Rollen wie Werkzeug–Ziel–Hindernis.
Zunächst bereitgestellte Merkmale isolieren die Repräsentationsfrage; Pixel und
weitere Modalitäten folgen als eigener Vergleich. Erfolg mit Merkmalen belegt
keine gelernte Wahrnehmung oder vollständige Agentenfähigkeit.

**Episode:** Supportevidenz → erschlossener Wissenszustand → zurückgehaltene
Abfragen → neue Evidenz/Korrektur → erneute und unbetroffene Kontrollabfragen.
Initiales Training lernt die Verfahren; Evaluation friert Gewichte und trainierte
Buffer ein. Die dynamische Evidenz und der Wissenszustand dürfen sich verändern.
Eine bekannte Abfrage kann das relevante Kriterium vorgeben; dessen selbständige
Entdeckung ist ein separater, schwierigerer Test. Bei mehreren mit der Evidenz
vereinbaren Regeln werden Hypothesen oder Unsicherheit bewertet; verborgenes
Generatorwissen wird dem Modell nicht als eindeutig erschließbar zugeschrieben.

**Vergleich:** (A) ein gemeinsames Zustandsformat und gemeinsame Kernoperationen;
(B) dasselbe Format mit spezialisierten Kernoperationen als Diagnose;
(C) passende spezialisierte Formate/Verfahren; (D) Beispielabruf mit vergleichbar
leistungsfähigem Leser. Ein Prototyp ist eine zusätzliche Kategoriereferenz,
kein künstlich schwacher universeller Gegner. Gleiche Evidenz, Splits und Such-
bzw. Trainingsbudgets; Unterschiede in Parametern, Trainingsbeispielen pro Familie,
Speicherbytes, tatsächlichen Evidenzzugriffen, FLOPs und Latenz separat ausweisen.
Kapazitätskurven ergänzen einen einzelnen Budgetpunkt. Zunächst Code-only-Zugriff
zur Diagnose, danach gleiche begrenzte Evidenzabrufe für alle Kandidaten.

**Übertragung:** Neue Instanzen, neue Kombinationen, zurückgehaltene Regeln innerhalb
trainierter Familien und vollständig zurückgehaltene Familien getrennt berichten.
Split nach generativer Regel/Struktur, nicht bloß nach Bild. Kennungen, Oberflächen
und Reihenfolge variieren; Testgenerator und Aufgabenlösung unabhängig prüfen.
Gemeinsames Training wird zusätzlich gegen Training pro Familie verglichen, um
Interferenz zu erkennen. Kombinierte Abfragen prüfen die Nutzung mehrerer gelernter
Abstraktionen zusammen; sie sind keine Voraussetzung für jede einzelne Kategorie.

**Kontrollen und Messung:** Wissenszustand entfernen/vertauschen, Supportevidenz
entfernen und gezielt widersprechende Evidenz liefern. Ein Zustandswechsel soll
vorhersagbare Änderungen auslösen. Ein passender gespeicherter Zustand soll nach
Kontextwechsel abrufbar bleiben; Korrektur soll betroffene Antworten verbessern
und unbetroffenes Wissen erhalten. Trefferquote/Fehler je Familie und Neuheitsstufe,
Lernkurve über Supportumfang, Korrekturgewinn, Erhalt und Ressourcen getrennt
berichten, über unabhängige Seeds und Episoden mit Unsicherheitsintervallen.

**Entscheidung:** Vor dem Lauf absolute Mindestqualität und eine tolerierte Lücke
zum Spezialisten je Familie festlegen (z. B. drei Prozentpunkte als zu diskutierende
Nichtunterlegenheitsgrenze, nicht als bereits beschlossenes Gate), ebenso Seeds,
Populationen und Rechenbudget. Ein guter Gesamtmittelwert darf keine schwache
Familie verdecken. Bestehen stützt die getestete gemeinsame Lösung im benannten
Umfang. Scheitern isoliert zunächst Format, Leser, Training, Kapazität oder
Wahrnehmung; es widerlegt keine universelle Möglichkeit gemeinsamer Repräsentation.

Stehende Prinzipien: Quellen einmal vorbereiten und mehrfach nutzen; Details und
komprimierte Hypothesen getrennt zugänglich halten; Evidenzbesitz und Laufzeitwissen
explizit behandeln; überprüfbare Aufgabenresultate statt bloßer Embeddingähnlichkeit
verwenden. Spezialprojektionen sind eine Vergleichsoption, keine vorweggenommene
Lösung. Multimodale Rohdaten und adaptive Rechentiefe folgen nach dem kleinen Test,
um Wahrnehmungsfehler und zusätzliche Suche nicht mit Formatgüte zu vermischen.

## Mathematische Konkretisierung des Vergleichs

Alex bittet um tatsächliche Diskussion mit Claude Opus 5.5 bei `max` und Formeln,
die eine eindeutige Implementierung ermöglichen. Die [Spezifikation](shared-abstraction-spec.md)
definiert Eingabegrammatik, Tensorformen, gemeinsamen Erschließer/Leser,
gewichteten Loss, Vergleichsarme, Splits, Oracle, Korrektur und Entscheidungskriterien.
Die dortigen Reviewbelege trennen gemeinsame Schlussfolgerungen von methodischen
Präferenzen. Ein endlicher Generator und zufällig initialisierte Tensorpfade wurden
geprüft; kein Modelltraining und kein allgemeiner Abstraktionsnachweis.

## Selbst entdeckte Konzepte über die Zeit

27./28. September 2026. Alex fragt, ob die Architekturidee von
[Arc2Face](compact-instance-memory-ideas.md) (ein kompakter Code steuert einen
generativen Prior) auch Konzepte wie „Box“ lernen kann, und präzisiert: Am besten
lernt das Modell über die Zeit selbst, was Konzepte sind; „Box“ war nur ein Beispiel.
Die folgende Einordnung ist ein Diskussionsvorschlag des Assistenten (Claude Opus
5.5), gegengelesen von Fable 5.1 und Codex gpt-6-astra; keine beschlossene Methode,
kein Ergebnis und keine Opus-`max`-Review wie in den Abschnitten oben.

**Rolle eines code-konditionierten Generators.** Ein auf einen gegebenen Code
konditionierter Generator (Arc2Face; [Textual Inversion](https://arxiv.org/abs/2208.01618),
Gal et al. 2022, das bei eingefrorenem Modell ein neues Token-Embedding optimiert)
entdeckt allein kein Gruppierungskriterium. Generative Modelle mit latenter Struktur
können dagegen Gruppierungen finden (Mischungsmodelle; Objektgruppierung durch
Rekonstruktion wie [Slot Attention](https://arxiv.org/abs/2006.15055) und DINOSAUR
in der Tabelle oben). Hypothese: Als primäres Lernsignal lenkt Pixelerzeugung eher
auf Aussehen als auf Nutzung und ist teuer; generatives Lernen bleibt möglich, aber
keine Voraussetzung. Nützliche Rollen des konditionierten Generators: Sichtbarmachung
im Debug-Modus („zeig, was du für X hältst“), ein zusätzlicher Konsistenztest und
später Vorstellung für Planung. Instanz- wie Konzeptzustände können Unsicherheit
tragen; der Unterschied liegt in der zulässigen Variation über Mitglieder, nicht in
einer festgelegten Punkt- oder Verteilungsform.

**Vorgeschlagener Entdeckungskreislauf.** Leitkriterium ist Vorhersagenutzen:
Eine Kategorie lohnt sich, wenn sie nicht beobachtete Merkmale neuer Fälle
vorhersagt (Anderson 1991, *The adaptive nature of human categorization*,
Psychological Review 98). Andersons Algorithmus ist eine Näherung einer
Dirichlet-Prozess-Mischung ([Sanborn, Griffiths & Navarro 2010](https://cocosci.princeton.edu/tom/papers/rationalapproximations.pdf)):
Jede Beobachtung wird einer bekannten oder einer neuen Kategorie zugeordnet, gesteuert
von einem Kopplungsprior. [Neural Clustering Processes](https://arxiv.org/abs/1901.00409)
(Pakman et al., ICML 2020) amortisieren diese Zuordnung mit einem Mengen-Netz, das
auf Stichproben eines Generators geclusterter Daten trainiert und im Test mit festen
Gewichten verwendet wird; das passt zum G/F-Vertrag. Latent-Cause-Modelle
([Gershman & Niv 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2862793/)) übertragen
dieselbe Idee auf Lernen mit Handlungen. Die Zuordnung hängt von der gewählten
Repräsentation, Likelihood und dem Prior ab; sie belegt allein keine nützlichen
Kriterien und ist in PATH-WM nicht implementiert.

1. *Vorschlagen:* Evidenz, die keine vorhandene Abstraktion gut erklärt, erzeugt
   einen vorläufigen Eintrag mit Belegen.
2. *Bewähren (Agentenkriterium):* Zur Laufzeit gibt es keine beschriftete
   Zurückhaltemenge. Der Agent kann nur prequentiell urteilen (erst vorhersagen,
   dann spätere Evidenz beobachten) oder über Posteriormasse. Entfernen, Vertauschen
   und Wiederherstellen sowie unberührte Abschlussfragen sind dagegen
   *Evaluatorkontrollen*; ihre Labels dürfen keine Einträge festschreiben.
3. *Korrigieren:* Aufteilen, Zusammenlegen oder Verwerfen als versionierte Revision
   mit Invalidierung abhängiger Zustände. Das erfordert eine Erweiterung des
   Belegvertrags der [Spezifikation](shared-abstraction-spec.md) (§7 kennt nur
   Hinzufügen/Ersetzen/Zurückziehen bei extern vergebener `concept_id`): Das Modell
   müsste Belege selbst Einträgen zuordnen. Belege bleiben abrufbar, sodass neu
   partitioniert werden kann.
4. *Überlappung:* Mehrere Einteilungen derselben Instanzen sind zulässig (vgl. die
   Kreuzkategorisierung oben, ebenfalls ein Dirichlet-Prozess-Modell).
5. *Handeln:* Eingriffe können Hypothesen trennen; in der Grammatik der Spezifikation
   wählt der Generator `u`, nicht das Modell, daher liegt dieser Schritt außerhalb
   der kleinsten Prüfung.

Der Kern bleibt das G/F-Gerüst: Belege → Zustand → gemeinsamer Leser. Ohne
vorgegebene Gruppierung wären mehrere explizite Zustände mit Zuordnung ein Kandidat,
ein gemeinsamer Zustand mit abfrageabhängigem Lesen ein anderer; beide ändern den
Tensorvertrag (`Z:[B,K,d]` pro Episode). Unüberwachtes Meta-Lernen
([CACTUs](https://arxiv.org/abs/1810.02334), Hsu, Levine & Finn, ICLR 2019)
konstruiert Trainingsaufgaben automatisch durch Clustern von Embeddings; der
Meta-Lerner selbst erhält dort gruppierte Episoden und belegt den hier gemeinten
Entdeckungsvertrag nicht. Die
Review unterscheidet bereits „vorgegeben versus selbst erschlossen“ als getrennte,
schwerere Prüfung.

**Grenze und Hybrid.** Selbst entdeckte Abstraktionen folgen dem Nutzen, nicht den
Grenzen menschlicher Begriffe. In Andersons Modell ist ein Name nur ein weiteres
vorhersagbares Merkmal. Entsprechend ist ein gelegentlicher Hinweis von Alex („das
ist auch eine Box“) ein billiger, korrigierbarer Beleg, kein Oracle, das fertige
Konzepte in den Graphen schreibt. Vorgeschlagen ist ein Hybrid: selbstständige
Entdeckung, Namen und Grenzfälle gelegentlich von außen.

**Stehende Prinzipien.** *Vorschlagen und Prüfen trennen* (Kandidat aus schlechter
Erklärung, Bestätigung prequentiell); *Zustand mit Zuständigkeit und Lebensdauer*
(vorläufige, bestätigte und verworfene Einträge mit Belegen); *vor dem Verwerfen
organisieren* (Belege bleiben abrufbar, Neupartitionierung möglich); *Rechenaufwand
nützlich verteilen* (harte Grenze für Kandidatensuche und -zahl); *das System
trainieren, das laufen wird* (Episoden ohne vorgegebene Gruppierung). Der erwartete
Nutzen ist, dass neue Abstraktionen ohne Gewichtsänderung und ohne vorgegebene
Gruppierung entstehen und korrigierbar bleiben. Kosten sind gehaltene gegenüber
gelesenen Belegen, Kandidatenzustände, Parameter, Such-/Aktualisierungsrechnung und
Latenz; Gefahr vieler kurzlebiger Einträge.

**Kleinste sinnvolle Prüfung (Vorschlag, nicht geplant).** In der endlichen Grammatik
der Spezifikation sind Belege `x` gleichverteilt; „ungruppiert“ kann dort nur heißen,
dass Belege mehrerer Regeln gemischt sind. Dann ist die Abfrage ohne Zusatz
mehrdeutig. Nötig wären: gemischte Belege aus demselben Split, ein abfrageseitiger
Hinweis (Ankerbeleg oder paarweise Frage „gleiche Regel wie dieser Beleg?“), ein
Mischungs-Referenzposterior in §6 (bei bedingter Unabhängigkeit der Belege gegeben
Regeln und Gewichte faktorisiert die Likelihood pro Beleg; exakte Aufzählung verlangt
zusätzlich ein kleines festes `k`, einen erklärten Mischungsprior und ein Budget für
die Aufzählung, etwa 280² Regelpaare mal Belege für k = 2) und vom Modell vergebene
Zuordnungen im Belegspeicher. Vergleich mit dem vorgegeben gruppierten Arm bei
gleicher Datenmenge und gleichem Budget, wobei die Gruppierung als privilegierte
Information ausgewiesen wird, sowie mit dem Mischungs-Posterior als
Informationsreferenz. Die hier besprochenen Verfahren belegen das vollständige
PATH-WM-Ziel autonomer Konzeptentdeckung nicht.

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
