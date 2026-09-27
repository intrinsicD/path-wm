# World Labs (RTFM, Atlas): mögliche Ideen für PATH-WM

27. September 2026. Auf Alex' Wunsch gesichtet: [Website](https://www.worldlabs.ai/),
[RTFM-Blog](https://www.worldlabs.ai/blog/rtfm), [Atlas-Blog](https://www.worldlabs.ai/blog/atlas).
Nur Blogposts, kein Paper, kaum Architekturdetails. **Nichts davon ist übernommen.**
Diese Datei hält mögliche Lösungen fest, bis ein konkreter Mangel sie rechtfertigt.

## Was dort beschrieben wird

- **RTFM:** autoregressiver Diffusions-Transformer über Videobildern, echtzeitfähig
  auf einer H100 (Destillation). Keine explizite 3D-Geometrie: Eingabebilder werden
  zu Aktivierungen im KV-Cache, die die Welt implizit darstellen; das Rendern ist
  gelernt. Jedes Bild trägt eine Pose. Für ein neues Bild werden die räumlich nahen
  gespeicherten Bilder als eigener Kontext abgerufen („context juggling“), statt
  über eine immer längere Bildfolge zu rechnen.
- **Atlas:** multimodaler autoregressiver Rectified-Flow-Transformer (Text, Bild,
  Tiefe, Kamerapose). Pose ist eine native numerische Eingabe, kein Text. Die Aufgabe
  (Rekonstruktion, neue Ansicht, Video, Splats) ergibt sich allein aus der Anordnung
  der Sequenz. Skaliert über mehrere Modellgrößen, von Grund auf vortrainiert.

## Mögliche Ideen und ihre Anschlussstellen

| Idee | Anschluss in PATH-WM | Status |
| --- | --- | --- |
| Beobachtungen als vorbereitete Encoder-Tokens/KV speichern, mit exaktem Schlüssel (Ort, Szene, Zeit, Instanz) nächste k abrufen | Offene „persistent detail retrieval“ im [Token-Budget-Plan](token-budget-plan.md#following-phases-and-scope) und [Encoder-Plan](encoder-token-budget-plan.md); prepare-once/reuse und KV-Wiederverwendung des festen Kontexts innerhalb einer Generierung existieren; Abruf über Beobachtungen hinweg nicht | nicht geplant |
| Exakte Metadaten zusätzlich als gelernte Eingabe des Kerns, nicht nur als Verwaltungsreferenz | Zielvorgabe „exakte Quellreferenzen begleiten latente Inhalte“ ([Ziel](integrated-latent-agent-goal.md)) | nicht geplant |
| Eine Sequenzschnittstelle, Aufgabe nur durch Anordnung; AR über Elemente, Diffusion innerhalb eines Elements | Alex' Hypothese latenter Diffusion ([Ziel](integrated-latent-agent-goal.md#mechanismen-und-offene-vergleiche)) | nur Beleg in großem Maßstab |
| Generativer Decoder statt reiner MSE-Rekonstruktion für realistische Details | [Rekonstruktions-Review](reconstruction-bottleneck-review.md) behandelt perzeptuelle/adversariale Ziele | siehe unten |

## Abgrenzung

Der [Kontextabruf-Plan](context-retrieval-plan.md) ist **nicht** dieselbe Idee: Er
wählt lexikalische Schlüssel aus und kopiert exakte symbolische Nutzlasten (≤32
gescannte Schlüssel, ≤2 gelesene Nutzlasten). Wahrnehmungsdetails als abrufbare
Aktivierungen deckt er nicht ab. Seine Verträge (Versionen, Veralten, Budgetgrenzen,
Neustart) wären aber die Vorlage für einen Detailspeicher.

Die gesichteten RTFM-Texte beschreiben weder gezielte Korrektur oder Invalidierung
des impliziten KV-Gedächtnisses noch Konzepte. Das stützt die bestehende Trennung: Wahrnehmungsdetails
dürfen in einem abrufbaren Cache liegen; Konzepte, Instanzen und Korrekturen gehören
in den versionierten Graphen. Ein Detailcache muss an Quellversion und Gewichte
gebunden sein (kein Caching über Gewichtsänderungen hinweg).

## Photorealismus: Skalierung oder Architektur?

Beides; ein wesentlicher Faktor ist das **Lernziel**, nicht nur die Größe.
MSE-optimale Rekonstruktion liefert bei unsicheren Details den Mittelwert, also
Unschärfe (Wahrnehmung-Verzerrung-Abwägung, Blau & Michaeli 2018). Diffusion/Flow,
adversariale oder perzeptuelle Ziele ziehen stattdessen plausible scharfe Details —
auf Kosten exakter Treue. Skalierung und riesige Videodaten bestimmen dann, wie
*breit* (beliebige Szenen) und *konsistent* das gelingt. Schmale Domänen erreichen
Photorealismus mit deutlich kleineren Modellen (etwa StyleGAN für Gesichter).

Für PATH-WM heißt das: Realismus und Treue getrennt messen. Pixel-MSE bleibt das
Treuemaß; ein generativer Decoder bräuchte eigene Realismusmetriken und darf
erfundene Details nicht als erinnerte Evidenz ausgeben. Abruf gespeicherter
Beobachtungen (Zeile 1) liefert beobachtete statt erfundener Evidenz, wo die Szene
schon gesehen wurde; die Treue hängt weiter von Kodierung und Dekodierung ab.

### Alex' Ziel (27. September): genaue Erinnerung ohne gespeicherte Bilder

Festgehalten im [Ziel](integrated-latent-agent-goal.md#vom-nutzer-festgelegtes-ziel)
samt vorgeschlagener, mit Alex zu bestätigender Lesart. Messung mit einem
unabhängigen Identitätsprüfer (nur zur Evaluation, nie als Trainingsverlust; Spur A
des [Codec-Reviews](visual-codec-review.md)).
Werkzeuge und Architekturideen dazu: [Ideensammlung](compact-instance-memory-ideas.md).
Anschlussstellen: [Token-Budget-Plan](token-budget-plan.md#following-phases-and-scope),
[Kontextabruf](context-retrieval-plan.md#remaining-architecture-work), [latenter Kern](latent-core.md),
[Ziel](integrated-latent-agent-goal.md#mechanismen-und-offene-vergleiche), [Bildcode-Plan](image-code-contract-plan.md).

## Wann wieder aufgreifen

Wenn ein Feindetail-Task zeigt, dass Details nach Kompression im Zustand fehlen,
aber in gespeicherten Beobachtungen vorhanden wären; oder wenn Bildausgabe Realismus
statt Treue verlangt. Dann gemeinsam einen Plan mit Kapazität, Eviction, Schlüsseln
und budgetgleichen Kontrollen aufstellen.
