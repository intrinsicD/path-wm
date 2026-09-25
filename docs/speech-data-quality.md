# Sprachdaten und Gesprächsqualität

25.September2026: Alex fragt nach vorhandenen Datensätzen und ausreichender
Sprachqualität für Gespräche. Quellenprüfung, keine Downloads von Audiodaten oder
neuen Trainings; folgende Auswahl bleibt ein Vorschlag zum [Sprachplan](grounded-speech-plan.md).

| Quelle | Konkreter Nutzen | Grenze |
| --- | --- | --- |
| [Speech-MASSIVE](https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE) | Deutsche Aufnahmen mit Absicht/Argumenten; de-DE train11514 und validation2033 per Viewer-API bestätigt | Einzelne Aufträge, keine Verknüpfung mit unserer Szene; CC-BY-NC-SA-4.0 |
| [Common Voice](https://github.com/common-voice/cv-dataset/) | Unterschiedliche Stimmen und Aufnahmen; gelesene und spontane Sprache | Transkriptionsdaten liefern nicht automatisch Dialog-/Weltverständnis; deutschen Umfang pro Version prüfen |
| [Multilingual LibriSpeech](https://www.openslr.org/94/) | Größerer deutscher Audio-/Textbestand für Sprachgrundlagen, CC-BY4.0 | Gelesene Hörbücher, keine interaktiven Gespräche |
| [SpokenWOZ](https://spokenwoz.github.io/) | Aufgabenbezogene menschliche Dialoge und Gesprächszustände | Nicht als fertiger deutscher, szenengebundener Kommandodatensatz behandeln; CC-BY-NC4.0 |
| [Thorsten-Voice](https://www.thorsten-voice.de/ueber-das-projekt/aufnahmen/) | Saubere deutsche Audio-/Textpaare für Sprachausgabe; neutral und dargestellte Ausdrucksvarianten, CC0 | Eine Stimme ist keine Sprecherrobustheit; gespielter Ausdruck ist kein Beweis innerer Emotion |

Die Speech-MASSIVE-Karte nennt den Testsplit separat unter
[Speech-MASSIVE-test](https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test).
Das erklärt sein Fehlen in der Haupt-Viewer-API. Der115-Beispiele-Split ist ein
Few-shot-Angebot, keine zusätzliche unabhängige Testpopulation. MASSIVE ohne
„Speech“ ist der Textdatensatz; keine vorhandenen Audioaufnahmen daraus ableiten.
Die API-Metadaten wurden gelesen; es wurden keine Audiozeilen/Testantworten geladen.

Empfehlung: Speech-MASSIVE-de für ersten Absicht-/Argumentvergleich, zusätzliche
eigene Audio-Szene-/Dialogpaare für Referenzen/Korrekturen, CommonVoice/MLS für weitere
akustische Variation nach Bedarf. Thorsten für Ausgabe separat prüfen. Keine Quelle
liefert allein die gemeinsame Sprach-, Szenen-, Erinnerungs- und Handlungskompetenz.

## Vorgeschlagene Qualitätsprüfung

1. **Verstehen:** exakte Absicht+Referent+Wert, Negationen, Zahlen, Korrekturen und
   passende Rückfragen. WER kann Transkriptionsfehler diagnostizieren, ersetzt aber
   keine semantische Bewertung. Der direkte Audiopfad braucht kein Laufzeittranskript.
2. **Dialog:** Aufgaben-/Antwortkorrektheit über mehrere Turns, Erinnerung an Bezüge,
   Korrekturverarbeitung, unbekannte Inhalte und ausbleibende/übermäßige Rückfragen.
3. **Ausgabe:** unabhängig zurücktranskribiertes Audio gegen beabsichtigten Wortlaut,
   Auslassungen/Wiederholungen, Zahlen/Namen und hörbare Artefakte. Blindhörtests mit
   mehreren Deutschsprechenden für Verständlichkeit, Natürlichkeit und Betonung,
   verglichen mit menschlichen Aufnahmen und einer festgehaltenen Referenz.
4. **Interaktion:** Ende der Nutzersprache bis tatsächlich hörbarer Antwort,
   Unterbrechung bis Wiedergabestopp, falsche Turn-Enden, überlappende Sprache und
   Audiopuffer-Aussetzer. Median und langsame Fälle getrennt berichten.
5. **Unabhängiger Test:** ungesehene Sprecher/Formulierungen, reale Mikrofone,
   Hintergrundgeräusche und längere Dialoge; Test nicht zur Optimierung verwenden.

[Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval) zeigt objektive
WER-/Sprecherähnlichkeitsprüfung. Der [zugehörige Bericht](https://arxiv.org/abs/2406.02430)
verwendet auch subjektive Bewertung. Diese Methodik kann Orientierung geben; deren
Benchmarkwerte sind keine deutschen Gesprächsqualitätsgrenzen für unser Modell.

Abnahmeschwellen, Testumfang und Referenz werden vor einem Lauf festgelegt. Kein
einzelner Loss oder automatischer Natürlichkeitsscore garantiert angenehme Gespräche.
Die bisherigen kurzen geplanten Trainingsscreens prüfen eine schmale Anbindung;
sie versprechen keine freie Unterhaltung. Für hohe Qualität ist die gezielte Nutzung
vortrainierter Sprachfähigkeiten eine zu prüfende Option, keine bereits ausgewählte
Abhängigkeit oder garantierte Abkürzung. Kein neuer Qualitätslauf wurde ausgeführt.
