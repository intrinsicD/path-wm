# Bedarfsgesteuerte Neural Engine für Inferenz

22. September 2026, Diskussionsvorschlag nach der GPU-Budgetklärung.
Alex stellt sich eine zentrale Neural Engine vor, die bereits trainierte Modelle
bei Bedarf nutzt, vergleichbar mit einem Buffer beim Rendern. Er grenzt dies
ausdrücklich auf Inferenz ein; das Training ist noch offen. Die undeutlich
transkribierten „GP-Punkte“ werden nicht als festgelegtes Speicherformat oder
bestätigte Graphknoten interpretiert. Die bedarfsgesteuerte Ausführung lässt sich
unabhängig davon beschreiben. Keine Implementierung beauftragt oder validiert.

## Vorgeschlagene Ausführung

```mermaid
flowchart LR
    S[Versionierte Modelle auf SSD] --> R[Begrenzter RAM-Vorrat]
    R --> G[GPU: aktive Gewichte und Arbeitsspeicher]
    Q[Aufgabe und aktuelle Beobachtung] --> E[Ausführungsplanung mit Budget]
    E --> G
    F[Gemeinsame Quellmerkmale] --> G
    G --> O[Ergebnis und erforderlicher Sitzungszustand]
    O --> W[Zeitgestempelte Evidenz im vorhandenen Speicher]
```

Ein kleines häufig benötigtes Grundmodell kann resident bleiben. Zusätzliche
Leser bzw. kompatible trainierte Module werden aus RAM vorgeladen oder bei Bedarf
geladen, ausgeführt und bei Speicherdruck wieder entfernt. Häufig wiederverwendete
Module über mehrere Frames behalten; ständiges Laden pro Frame würde gerade die
Geschwindigkeitspriorität gefährden. Ein handnahes Objekt könnte beispielsweise
die Handanalyse auslösen, deren Ergebnis danach als Beobachtung verfügbar bleibt.
Welche Leser nötig sind, ist eine fachliche Auswahl; ihre Speicherplatzierung und
Aufrufreihenfolge sind Aufgaben der Laufzeitverwaltung. Zunächst explizite Regeln,
später eine gemessene Policy als Vergleich. Die vorhandene Kontext-TaskPolicy
implementiert damit nicht automatisch Modell- oder GPU-Residenzsteuerung.

Die GPU enthält unterschiedliche Dinge mit unterschiedlichen Lebensdauern:

- Trainierte Gewichte, identifiziert durch Modell-/Checkpointversion und Präzision.
- Gemeinsame Quellmerkmale, nur bei identischer Quelle, Vorverarbeitung,
  Encoder-/Konditionierungsversion wiederverwendbar.
- Temporäre Aktivierungen und Workspaces eines Aufrufs.
- Erforderliche fortlaufende Sitzungszustände, die vor Entladen erhalten werden
  müssen; Modell-Cache-Eviction darf keine Person oder zeitliche Evidenz löschen.

Der Knowledge-Graph kann auf Modelle und Resultate verweisen. Gewichte müssen
nicht je Entität kopiert oder als Graphdaten bei jedem Aufruf neu gelesen werden.
Die akzeptierten geteilten Multiskalenmerkmale mit privaten Verbrauchern bleiben
kompatibel: nur kompatible Schnittstellen erlauben Feature-Wiederverwendung.
Externe Spezialisten mit eigenen Eingängen/Backbones teilen diese nicht automatisch.
Einzelne Filter eines dichten Netzes sind ohne deklarierte Abhängigkeiten keine
beliebig austauschbaren ausführbaren Einheiten. Zuerst komplette kleine Module
oder definierte Blöcke betrachten; trainiertes Routing/Sparsity ist eine weitere
Modellentscheidung und nicht durch Puffern allein gegeben.

## Speicher, Latenz und überprüfbarer Umfang

Die gesamte Bibliothek darf größer sein als VRAM. Gleichzeitig aktive Gewichte,
Aktivierungen, aktuelle Features/Zustände, Kopierpuffer und CUDA-Workspaces müssen
in das vorgeschlagene 6-GiB-Prozessbudget passen. Vorladen kann kurzzeitig beide
Module resident halten; diesen Peak mitzählen. Ein einzelner zu großer Aufruf
braucht ein kleineres Modell oder eine explizite Block-/Offload-Aufteilung.
Ein Buffer erhöht weder Rechenleistung noch Transferbandbreite.

[Accelerate Big Model Inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
zeigt bestehenden CPU-/GPU-/Disk-Offload. Das belegt den Mechanismus, keinen
automatischen dynamischen Spezialistenplaner oder Echtzeitbetrieb von PATH-WM.
[PyTorch CUDA-Semantik](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management)
trennt lebende Tensoren und ungenutzten Allocator-Cache: `empty_cache()` entfernt
keine noch referenzierten Modellgewichte. Bei asynchronen Kopien/Aufrufen müssen
Abhängigkeiten und Lebensdauer bis zum Abschluss gesichert sein. Parallelität
und Prefetch sind zu messen; zunächst einfache sequentielle Planung vergleichen.

Prinzipien: gemeinsame Vorbereitung wiederverwenden; Evidenz, Modellwissen und
verwerfbare Caches getrennt besitzen; bedingte Detailarbeit; Speicherverkehr und
End-to-End-Latenz statt nur Modell-FPS messen. Keine neue allgemeine Registry oder
Frameworkhierarchie nötig: ein begrenzter expliziter Aufrufplan in der vorhandenen
Recipe-/Bibliotheksstruktur reicht als erster Vergleich.

Kleinster späterer Test: zwei fest gewählte trainierte Leser mit realer
Aufruffolge; wiederholte Treffer und wechselnde Anforderungen. Gleichbleibende
Modelle/Eingänge/Präzision vergleichen, einmal mit ständigem Neuladen und einmal
mit begrenztem Verbleib im GPU-Speicher; All-resident nur falls es passt.
Separat Peak-VRAM, CPU-RAM, Transferbytes, Kalt-/Warmlatenz und p50/p95 messen;
Output-/Zustandsübereinstimmung bei unveränderter Numerik mit vorab gesetzter
Toleranz prüfen. Quellenzugriff und Zeitkausalität erhalten. Prefetch erst als
eigenen Faktor ergänzen. Modellwahl, konkrete Gates und Trainingsverfahren offen.

Training würde zusätzlich Gradienten, Optimizer, Aktivierungen und konsistente
Gewichtsversionen benötigen. Die Inferenzidee legt weder getrenntes Spezialisten-
training noch gemeinsames End-to-End-Training fest; die bisherige gemeinsame
Lernrichtung wird durch diese Speicher-/Ausführungsdiskussion nicht ersetzt.

## Ein eintreffender Frame: konkreter vorgeschlagener Datenfluss

22. September, Folgefrage nach dem genauen Ablauf im Modell. Dies verbindet die
akzeptierte Multiskalenrichtung mit der vorgeschlagenen Personen-/Objektwahrnehmung;
keine bereits vollständig implementierte Echtzeitpipeline und kein festgelegter Takt.

| Schritt | Rechenoperation / Zuständigkeit | Ergebnis |
| --- | --- | --- |
| 1. Eingang | Aufrufer erhält Bildtensor, Quellen-ID, Aufnahme-/Verfügbarkeitszeit; prüft Format, Gültigkeit und Koordinatenabbildung der Vorverarbeitung. | Eindeutiges Beobachtungspaket. |
| 2. Vorhersage | Dynamik liest vorigen Zustand, Zeitabstand und gegebenenfalls bekannte eigene Aktion. | Erwarteter Zustand vor der Bildkorrektur, nur aus bisheriger Evidenz. |
| 3. Gemeinsames Encoding | Pro Skala R → lernbare Filterbank B → Nachverarbeitung P; f nach P für Leser behalten, C für die nächste Skala. | Multiskalenmerkmale F samt Ort, Zeit und Gültigkeit; keine zwingende Pixel-Klassenmaske. |
| 4. Objekt-/Zustandslesen | Aktive Verbraucher lesen F mit Abfragen/Transformer-Schleifen und eigenen Ausgabeschichten. Kandidaten mit aktuellen Merkmalen plus erwartetem Ort/Verlauf zuordnen. | Objektkandidaten, Eigenschaften, Lage/Bewegung, revidierbare Zuordnungen und neue/ungeklärte Fälle. |
| 5. Bedingte Verfeinerung | Aufgabenbedarf, Sichtbarkeit oder Widersprüche lösen Detail-Leser aus; Engine sichert Gewichte/Arbeitsraum und liest passende Skalen bzw. Quellausschnitte. | Beispielsweise Handpose, Mundbewegung oder Objektmaske mit Quellenbezug; keine Pflicht für alle Objekte/Frames. |
| 6. Korrektur und Ereignislesen | Evidenz gegen Prior abgleichen; Beziehungen/Zustände aktualisieren und aus Verlauf plus Beobachtung Ereignisse schätzen. | Welt-/Entitätszustand, etwa gemeinsame Bewegung von Tasse und Hand statt bloßer Nähe. |
| 7. Commit und Reaktion | Aufrufer veröffentlicht Zustands-/Evidenzänderungen einmal mit Herkunft/Zeit; TaskPolicy/Thinker liest bei Bedarf aktuellen Zustand und ausgewählte Historie. | Gedächtnis, Antwort oder Aktionsvorschlag bei entsprechendem Bedarf; kein Sprachoutput pro Frame nötig. |
| 8. Ressourcenpflege | Nicht mehr benötigte temporäre Tensoren freigeben; häufig benutzte Gewichte und deklarierte Zustände behalten. | Begrenzter GPU-Arbeitssatz; Historie bleibt erhalten. |

Vorhersage und Encoding können bei passenden Abhängigkeiten parallel stattfinden;
die Tabelle legt keine CUDA-Reihenfolge fest. Die Uhr/Aktion einmal je gültigem
Beobachtungsereignis fortschreiben, nicht für jeden zusätzlichen Leser erneut.
Langsame Detailergebnisse brauchen einen Vertrag für Aufnahme-/Verfügbarkeitszeit
und Korrektur: keine rückwirkend verfügbare Evidenz, keine doppelt gezählte Quelle.
Ein schneller Pfad darf nicht unbemerkt auf alle teuren Leser warten.

Beispiel, nur gewünschtes Verhalten: Person P7 und Tasse C12 sind bekannte
Kandidaten. Im neuen Frame nähert sich P7s Hand C12; ein gezielter Hand-/Objektleser
prüft die Region. Erst ausreichender Verlauf stützt Aufnehmen oder Halten;
ein einzelner Kontaktverdacht beweist das nicht. Evidenz und geschätzte Zustände
erhalten Zeitbezüge, stabile Körperform wird nur bei ausreichender neuer Evidenz
aktualisiert. Ohne Detailanalyse bleibt der entsprechende Zustand ungeklärt.

Inferenz verändert Aktivierungen, private Leserzustände, Weltbelief und Speicher.
Gewichte der trainierten Filter/Leser bleiben dabei unverändert. Online-Lernen
wäre eine eigene Operation mit Budget und Versionierung, kein Schritt je Frame.

Code-Abgleich: `BeliefAgent.begin_event` ruft die Dynamik einmal für den Zeitabstand
auf; `add_packet`/`correct_packets` korrigieren Beobachtungen und `commit_event`
schreibt in das Sitzungsgedächtnis. Die aktuelle Korrektur aktualisiert kategoriale
Verteilungen, nicht nochmals `h`; siehe [latenter Kern](latent-core.md).
`WorldSession.observe` nimmt heute gelieferte Kandidaten und Pakete entgegen und
veröffentlicht gebündelten Zustand. Automatische natürliche Bildkandidaten, die
neue R/B/P/C-Schnittstelle, alle Personenspezialisten und GPU-Residenzplanung sind
damit nicht implementiert. Grundlagen und Entwurf getrennt halten.

Prinzipien: gleiche Framequelle für kompatible Leser vorbereiten, private Abfragen
und Zustandslebensdauern trennen, Details gezielt lesen, Evidenz/Inferenz/Cache
unterscheiden. Kleinster künftiger Nachweis: begrenzte Interaktion mit tatsächlichem
Zustand-/Gedächtnisreadout, einmaliger Zeitfortschreibung, Zuordnung, verfügbaren
Quellen und Vollpfad-Latenz/Peak-VRAM. Keine neue Test-/Trainingsausführung hier.
