# Gesprochene Dialoge und Kommandos: nächster Forschungsplan

25. September 2026. Alex beauftragt die gemeinsame Planung mit tatsächlichem Claude
für interaktive Gespräche, gesprochene Mitteilungen und Kommandos, einschließlich
Bezügen zu anderen Modalitäten. **Planung, keine Freigabe neuer Sprachtrainings oder
Modellinstallation.** Direkte latente Anbindung bleibt das Ziel; eine Textkaskade
ist kein verpflichtender Laufzeitpfad. Die Gesprächsdiskussion steht in
[agent-voice-design.md](agent-voice-design.md).

## Entscheidungsvorschlag

Zuerst prüfen, ob gesprochene Fragen und Aufträge den bestehenden gemeinsamen Zustand
korrekt nutzen. Die erste Ausgabe darf ein prüfbares Antwort-/Zielrecord sein.
Sprachausgabe kann parallel vorbereitet werden. Gute Audiorekonstruktion ist ein
nützlicher Test für erhaltene akustische Details, **keine Voraussetzung oder ein
Nachweis für Sprachverständnis**. Diese Präzisierung ersetzt die anfängliche mündliche
Reihenfolge „erst Codec vollständig, dann Verständnis“.

Beginnen mit vollständigen Äußerungen, aber zeitliche Unterstützung, Segmentgrenzen
und einen später prüfbaren kausalen Zugriff bereits im Datenvertrag festlegen.
Vollständige Äußerungen beweisen kein Streaming. Das bestehende Modell und seine
wirklichen Komponenten bleiben Gegenstand aller Versuche.

## Was vorhanden ist und was konkret fehlt

| Zuständigkeit | Vorhanden | Fehlende Verbindung / Lernfähigkeit |
| --- | --- | --- |
| Audioeingang | `MultiScaleAudioEncoder`: Wellenform-Patches, Zeit, Gültigkeit, Hierarchie | Auf echter Sprache trainierte semantische Anbindung; Sample-Rate-/Zeitvertrag; kausales Streaming und Cache-Äquivalenz sind unbewiesen |
| Auftrag | `TaskRequest.instruction` und `BeliefAgent.task_tokens` lesen derzeit Text | Audiofeatures plus Quellen-/Versionsbindung als alternativer Instruktionseingang; kein heimlicher Transkriptkanal |
| Gemeinsame Verarbeitung | `TaskInterpreter`, Kern, Zustand, Speicherzugriff | Gelernte Zuordnung von Sprache zu Instanzen, Eigenschaften, Absicht und Kontext; allgemeine Sprachkompetenz fehlt als Nachweis |
| Steuerung | `TaskPolicy`: think/recall/imagine/act/emit/ask/finish | Evidenzabhängiges Zuhören, Nachfragen und Antworten; das vorhandene `ask`-Signal formuliert noch keine Rückfrage |
| Kommandos | `GoalSpec`, `TypedAction`, Prüfroutinen im vorhandenen integrierten Agenten | Gelernter, überprüfbarer Sprache-zu-Ziel-Leser; keine beliebigen Toolaufrufe |
| Gesprächsgeschichte | `WorldSession`/`EpisodeClient`: Quellen, Korrektur, abgeleitete Chunks, Abbruch | Audiosenke mit Abspieluhr, Echo-/Sprecherbehandlung und trainierte Dialogpolitik |
| Audioausgabe | `AudioDecoder` liest Zustand und erzeugt einen Block fester Länge | Kompatibler zeitlicher Sprachgenerator, Stopp-/Dauerentscheidungen, verständliche Sprache und Streaming-Wiedergabe |

Code: [Audioencoder](../pathwm/models/multiscale.py), [Aufträge](../pathwm/models/tasks.py),
[Agent](../pathwm/models/agent.py), [EpisodeClient](../pathwm/world_state/episodes.py),
[Ziele/Aktionen](../pathwm/world_state/unified.py), [Audiodecoder](../pathwm/models/modalities.py).
Die bestehenden [Aktionsregeln](action-semantics-design.md) und
[Ergebnisprüfung](decision-design.md) bleiben maßgeblich.

## Ein gemeinsamer Zustand, getrennte Verantwortlichkeiten

Audioabschnitte liefern Features und Quellenzeiten. Der Kern verknüpft sie mit
Beobachtungen und Erinnerungen. Vorläufige Interpretationen sind abgeleiteter Zustand;
eine spätere Wortfolge oder Korrektur kann sie ändern. Quellenaufnahme und Behauptung
bleiben getrennt: „Person sagt X“ beweist nicht, dass X in der Welt stimmt.

Eine Steuerung unter der bestehenden TaskPolicy unterscheidet:

- **Warten:** Für die Entscheidung fehlt noch erwartbare Eingabe.
- **Denken/Abrufen:** Vorhandene Informationen können die Unsicherheit auflösen.
- **Nachfragen:** Referent oder erforderliche Information bleibt mehrdeutig/fehlend.
- **Antworten/Handeln:** Der konkrete Inhalt und die Voraussetzungen sind ausreichend
  geklärt; eine spätere Ergebnisprüfung bleibt eigenständig.

Ein stabiler latenter Zustand oder ein hoher Completion-Score ist kein Ersatz für
diese Prüfungen. Anfangs liefern kontrollierte Episoden objektive Labels für
Mehrdeutigkeit/fehlende Argumente; kalibrierte Entscheidungen werden daran gemessen.
Ein einfaches Kommando darf schon im Eingabepfad leicht erkennbar sein. Erst bei
Aufgaben, die Szene oder Erinnerung benötigen, muss deren kausaler Einfluss sichtbar
werden. Es wäre falsch, aus einem guten Adapter-only-Intentklassifikator allgemein
einen unerlaubten Bypass abzuleiten.

## Kleinster sinnvoller Versuch: verstehen vor sprechen

Die vorhandene Welt kann Lampenzustandsziele behandeln. Deshalb zunächst Fragen wie
„Ist die linke Lampe an?“ und Ziele wie „Sorge dafür, dass die linke Lampe an ist“.
Die zweite Form ist ein **Ziel**, kein erfundener direkter Lampenschalter: Der bestehende
Ausführungspfad muss die tatsächlich erlaubten Aktionen bestimmen. Keine neue
Objektbewegung oder andere unimplementierte Handlung als Ersatzwelt hinzufügen.

Spätere Episoden desselben Tests ergänzen „und die andere?“, eine eindeutige vorherige
Referenz und „nein, ich meinte die andere“. Mehrdeutige, abgebrochene und außerhalb des
Schemas liegende Aufträge müssen Nachfragen/Unbekannt erzeugen, keinen geratenen Aktoraufruf.

Der erste fehlende Softwarevertrag ist ein Audio-Instruktionsfeature-Eingang in den
bestehenden Aufgabenpfad samt gelerntem Ziel-Leser. Er muss Quellenreferenz, Gültigkeit,
Zeit und Repräsentationsversion erhalten. Eine konkrete API-Skizze und ihre Tests kommen
vor einer Implementierung. Danach native Gradienten-, Quellen-, Korrektur- und
Resume-Prüfung. Kein Oracle-Goal darf als angeblich gelernte Sprachleistung gelten.

Vorgeschlagener Vertrag ohne neue Speicherinstanz: Instruktionseingang = `turn_id`,
Quellenreferenz/Hash, `FeaturePyramid`/gültige Tokens, Quellen- und Verfügbarkeitszeit,
Repräsentations-/Checkpointversion, Abschluss-/Revisionsstatus. Ziel-Leser = Frage
oder Lampenzustandsziel, gebundene Instanzreferenz, Wert und Status
`resolved / ambiguous / missing / unsupported`; ungelöste Argumente bleiben offen.
Klärung = Grund und benötigtes Argument, zunächst als prüfbarer Record. Die gesprochene
Formulierung dieser Klärung ist eine spätere Ausgabefähigkeit. Keine neue API ist
hier bereits implementiert.

Trainingsziele: richtige Frage/Absicht, tatsächlich gemeinte Instanz, Zustandswert,
Antwortwert oder begründete Rückfrage. Gegebenenfalls ein Hilfssignal aus Transkripten;
zur Laufzeit erhält der Audioversuch weder Transkript noch verborgenes Ziel. Nach einem
gelungenen Ziel-Leser kann der vorhandene Ausführer angeschlossen werden. Zielerkennung,
Aktionswahl und beobachteter Erfolg werden separat bewertet.

### Daten und Kontrollen vor dem ersten Qualitätslauf festlegen

Deutsch, zunächst eine sprechende Person pro Aufnahme; mindestens vier Trainings-,
zwei Entwicklungs- und zwei unberührte Testsprecher als vorgeschlagener kleiner Screen.
Testaufnahmen sind echte Sprache. Synthetische Stimmen können Training ergänzen;
ungesehene TTS-Stimmen ersetzen keine realen Testsprecher. Noch keine Quelle ausgewählt
oder Sammlung begonnen. Daten-/Nutzungsrechte und die exakten Aufnahmemanifeste sind
Voraussetzungen der späteren Ausführung, nicht heute bereits erfüllte Bedingungen.

Trennung nach Aufnahme, Sprecher, Formulierungsfamilie, Szene und Referenzkombination;
keine Segmente derselben Aufnahme in verschiedenen Splits. Absicht/Ziel dürfen nicht
mit Stimme, Pausenlänge, TTS-Stil, Objektposition oder Hintergrundgeräusch korrelieren.
Entwicklung und finaler Test bleiben getrennt. Akustische Bedingungen zusätzlich
aufschlüsseln; ein kleiner Screen belegt keine allgemeine Robustheit.

Kontrollen: vertauschte Audios, Stille, entfernte Szene, entfernte Vorgeschichte,
entfernte Korrektur sowie einfache Label-/Audio-only-Baselines. Bei Aufgaben mit
notwendigem Kontext: identisches Audio bei geänderter Szene oder Erinnerung muss die
korrekte Zuordnung ändern. Unveränderte irrelevante Daten dürfen die Antwort nicht
ändern. Ein Transkript durch denselben Kern ist ein **empirischer Vergleich**, keine
mathematische Obergrenze. Ein geliefertes korrektes Ziel prüft den nachgelagerten
Ausführer gesondert. Wenn Audio und Text scheitern, sind Daten, Training und Anbindung
ebenfalls mögliche Ursachen; Kernkapazität ist damit nicht bewiesen.

Vorgeschlagene, vor Ausführung zu registrierende Entwicklungsschwellen: zwei Seeds;
mindestens95% exakt richtige Absicht+Instanz+Wert auf eindeutig lösbaren gehaltenen
Episoden und mindestens10 Prozentpunkte Abstand zur stärksten passenden Negativkontrolle;
mindestens95% richtige Reaktion bei kontrolliertem Referenz-/Kontextwechsel. Auf je200
mehrdeutigen/abgebrochenen/ungültigen Kommandos keine unberechtigte Aktorausführung.
Zusätzlich Antwortabdeckung messen, damit ständiges Nachfragen nicht besteht. Alle
Nenner, Klassen, getrennten Split-Ergebnisse und binomialen Intervalle berichten;
null beobachtete Fehlhandlungen ist keine universelle Nullfehlerrate.

Für den vorgeschlagenen finalen Screen je200 eindeutig lösbare Episoden in den
getrennten Gruppen einfache Aufträge, Szenenreferenz und Vorgeschichte/Korrektur;
keine gute Gesamtquote als Ersatz für jede einzelne Gruppe. Mehrdeutigkeit wird
vorab anhand der tatsächlich verfügbaren sichtbaren/gespeicherten Evidenz gelabelt:
mehrere zulässige Referenten = ambiguous, fehlender notwendiger Referent/Wert = missing,
unvollständiger Turn = wait, außerhalb des Aktionsschemas = unsupported. Vollständige
Simulatorzustände dürfen die Auswertung, aber keine verborgene Information im
Audio-Forward liefern. Ein detailliertes Labelmanifest wird vor Datenerzeugung geprüft.

Erster geplanter Softwaretest: vollständige native Architektur, drei Updates plus
exakter Resume, maximal5min/6GiB. Qualitätsruns erst nach eingefrorenem Daten-/Modellvertrag:
vorgeschlagen2000Updates, zwei Seeds, höchstens15min/6GiB je Lauf, finaler Checkpoint,
keine nachträgliche Anpassung der Schwellen oder automatische Budgetverlängerung.
Lernrate, Batches und tatsächlich trainierbare Parameter müssen nach Wahl der
Audioanbindung vor dem ersten Lauf festgeschrieben werden. Diese Zahlen sind ein
Planangebot, keine aktuelle Trainingsfreigabe oder Aussage zur Erreichbarkeit.

## Parallel vorbereiten, anschließend verbinden

**Ausgabe:** Ein kompatibler Sprachsequenz-Generator liest den gemeinsamen Zustand.
Er bestimmt konkrete sprachliche Realisierung und zeitliche Einheiten; der
Akustikdecoder macht daraus Audio. Gleiche Tensorbreite bedeutet keine kompatiblen
Codes. Sample-Rate, Token-/Featuredefinition, Masken, kausale Unterstützung, Modellversion
und gelernte Verbindung müssen übereinstimmen. Stimme und Ausdruck sind zusätzliche
trainierte Konditionierungen. Freie Generierung auf eigener Historie prüfen, nicht
nur Vorhersagen mit vorgegebenen vorherigen Zieltoken.

**Streaming:** Erst deklarierte Chunk-/Lookahead-Regeln, dann Training und Prefix-Replay.
Ändert man Audio hinter Zeitpunkt t, dürfen frühere committed Ausgaben unverändert
bleiben. Der Test umfasst Encoder, Normalisierung, Segmentierung, Metadaten und Kern;
eine kausale Maske erst im Kern repariert keine zukünftige Information im Encoder.
Sprecheridentität, Sprachaktivität und Turn-Ende sind unterschiedliche Hypothesen.
Zunächst separater Mikrofon-/Ausgabestrom, inklusive Eigenstimme/Echo; Mehrpersonendialog
ist eine eigene Erweiterung. Sprechererkennung ist keine Authentifizierung.

**Ausgabehistorie und Text:** Pro Äußerung erzeugte, wartende, teilweise abgespielte,
abgespielte und abgebrochene Segmente unterscheiden. Die Abspieluhr bestimmt, was gehört
wurde. Dazu passender Text braucht gelernte/ermittelte Ausrichtung; unabhängige Decoder
garantieren keine gleiche Wortfolge. Anfangs Segment-, später Wortzeitmarken. Bei
Unterbrechung bleibt die tatsächlich abgespielte Historie erhalten; wortgenaues Kürzen
an einem beliebigen Audioframe darf ohne Alignment nicht behauptet werden.

**Weitere Bedeutungen:** Nichtsprachliche Geräusche gegen bekannte Ereignisse/Quellen
prüfen; unsichere Quellenzuordnung erhalten. Prosodie wie Tempo, Energie, Pausen und
Intonationsverlauf von vermuteter Emotion unterscheiden. Kontext-, Wortlaut- und
Prosodiewechsel getrennt testen. Bedeutungen übertragen sich auf Bild/Text/Handlung
durch trainierte gemeinsame Bezüge und Aufgaben, nicht allein durch gleiche Latentbreite.

## Sprachwissen und öffentliche Vorbilder

Ein Codec lehrt weder Sprachbedeutung noch freie Unterhaltung. Optionen sind eigene
Sprachvortrainierung, deklarierte Offline-Distillation oder kompatibel integrierte
vortrainierte Komponenten. Keine dieser Optionen ist hier gewählt. Ein externes
Dialogmodell darf nicht unbemerkt die gesamte Aufgabe lösen und dem eigenen Kern
als Erfolg zugerechnet werden; ein generelles Verbot integrierter vortrainierter
Sprachmodule ist aber ebenfalls nicht vereinbart.

[Moshi](https://arxiv.org/abs/2410.00037) zeigt getrennte Benutzer-/System-Audioströme
und zeitlich ausgerichtete Text-/Audioerzeugung; es startet von einem vortrainierten
Textsprachmodell. Der [offizielle Code](https://github.com/kyutai-labs/moshi) enthält
den Streamingcodec Mimi. Das begründet Mechanismen als Referenz, keine Übernahme
oder lokale Leistungs-/Speicherzusage.

[Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) beschreibt blockweise Eingaben und
einen Talker, der versteckte Thinker-Repräsentationen zur Audioerzeugung nutzt. Sein
Thinker erzeugt auch Text: Das Papier beweist keine vollständig textfreie Architektur
und keine fertige Verbindung zu unserem Kern. Beide Quellen wurden am25.September2026
geprüft; keine Behauptung, dies seien die neuesten oder lokal besten Modelle.

## Review, Grenzen und nächste Entscheidung

Tatsächlicher Claude Opus5.5 medium prüft eine abstrakte, öffentliche Aufgabenbeschreibung;
private Quellen, Code und Messwerte werden nicht exportiert. Lokale Codeprüfung
bestimmt die tatsächlichen Lücken. Exakte Briefs/Antworten und Metadaten:
`runs/reviews/grounded_speech_plan_20260925/`.

Übernommen: Eingabe/Verständnis und Ausgabe zunächst unabhängig prüfbar machen,
Kontextinterventionen, echte Sprach-Testaufnahmen, frühe Zeit-/Kausalitätsverträge,
eigene Ausgabegeschichte und die Grenze zwischen schmalen Kommandos und Sprachwissen.
Zurückgewiesen/präzisiert: automatische Bedeutungskompatibilität durch Zeitbasis,
Text als garantierte Obergrenze, jedes Adapter-only-Intent als Bypass, unbegründetes
Verbot vortrainierter Laufzeitmodule und unimplementierte Bewegungsaktionen im Ersttest.

Claude bestätigt diese Korrekturen in der zweiten Antwort und zieht die stärkeren
Aussagen ausdrücklich zurück. Übernommene Ergänzungen: einfache und kontextabhängige
Aufgaben getrennt auswerten, Mehrdeutigkeit vorab anhand verfügbarer Evidenz labeln,
gemeinsamen Segment-/Abspielvertrag vor paralleler Ein-/Ausgabearbeit festlegen.
Die zweite Sitzung erhielt die vollständig zitierten Korrekturpunkte, nicht den
privaten Projektkontext. Es bleiben keine methodischen Differenzen in diesem
begrenzten Plan; Datenauswahl, konkrete vortrainierte Komponenten, Ressourcenfit und
Trainierbarkeit sind weiterhin empirisch offen.

Nächste Entscheidung: den Audio-Instruktionsvertrag und das Datenset für den kleinen
Lampenzustands-/Referenztest auswählen und konkret freigeben. Vorher kein großer
Dialogtrainingslauf. Grundsätze: Quelle einmal vorbereiten, Details nicht blind verwerfen,
ein gemeinsamer autoritativer Zustand, Erzeugung und Prüfung trennen, unveränderliche
Quellen-/Versionsbindung und tatsächlichen Laufzeitpfad unter seinem Zugriff trainieren.
Weder das Planreview noch ein späterer Kommandoscreen etabliert freie Unterhaltung.
