# Kompakte Instanzcodes mit realistischer Ausgabe: Ideensammlung

27. September 2026. Auf Alex' Wunsch gesammelt, ausgehend von der
[World-Labs-Sichtung](worldlabs-review.md). **Nichts ist geplant oder übernommen.**
Die Sammlung hält Werkzeuge und Architekturideen fest, bis ein konkreter Mangel oder
eine gemeinsame Planung sie aufgreift.

**Einordnung (27. September):** Die Anschlussstellen stehen in den Owner-Dokumenten:
[Ziel](integrated-latent-agent-goal.md#vom-nutzer-festgelegtes-ziel),
[Instanz/Zustand](architecture-walkthrough.md#close-these-interfaces-before-choosing-more-mechanisms),
[Aggregation/Revision](entity-memory-design.md), [Auto-Decoder und
Gewichtsanpassung](latent-concept-learning-review.md#wie-lernen-ohne-laufendes-nachtraining-möglich-wird),
[zweiteiliger Code und Messung](image-code-contract-plan.md#model-controlled-diffusion-transformer-with-direct-bypass),
[externer Decoder als Vergleich](image-output-plan.md), [Evaluationsprüfer](visual-codec-review.md),
[Ansichtsrezept](integrated-architecture-plan.md) §19, [Stimmprofil](agent-voice-design.md).
Diese Datei ist kein Plan.

## Ziel (Alex)

Der Agent soll sich eine Instanz, zum Beispiel ein Gesicht, genau merken, ohne Bilder
zu speichern. Im Debug-Modus oder auf Anfrage soll trotzdem ein photorealistisches
Bild entstehen. Arbeitsdefinition: allgemeines Wissen (Konzept, Prior) steckt in
eingefrorenen Gewichten; pro Instanz wird nur ein kompakter Code gespeichert und zur
Laufzeit aus weiteren Beobachtungen verfeinert. Vorgeschlagene, mit Alex zu bestätigende Lesart: „genau“ heißt identitätstreu,
nicht pixeltreu. Was der Code nicht enthält, ergänzt der Prior plausibel.

## Speicherbedarf pro Instanz (Größenordnung)

| Code | Zahlen | fp16 |
| --- | ---: | ---: |
| ArcFace-/EdgeFace-Identitätsvektor | 512 | 1 KB |
| StyleGAN W / W+ (18 Ebenen) | 512 / 9216 | 1 KB / 18 KB |
| FLAME-Form (MICA) + Ausdruck | ~300 + ~100 | < 1 KB |
| face-vid2vid: Keypoints pro Bild | einige Dutzend | < 1 KB/Bild |
| Zum Vergleich: RGB-Bild 256² unkomprimiert | 196 608 | 384 KB |

Betrifft Identitätscodes pro Instanz; der feine Bildcode des
[Bildcode-Plans](image-code-contract-plan.md) ist ausdrücklich keine Speicherkompression.

## Kandidaten

„Geprüft“ = am 27. September an Repository oder Paper nachgesehen. Übrige Angaben
stammen aus Vorwissen und müssen vor Verwendung geprüft werden.

| Ansatz | Was er zeigt | Rolle für PATH-WM | Größe / Lizenz |
| --- | --- | --- | --- |
| [Arc2Face](https://github.com/foivospar/Arc2Face) (ECCV 2024), geprüft | SD 1.5 nur auf den 512-d ArcFace-Vektor konditioniert, trainiert auf WebFace42M; Erweiterungen: Ausdruck über FLAME-Blendshapes, Pose-ControlNet, LCM-LoRA für 2–4 Schritte | **Tool:** Debug-Renderer aus einem Identitätscode. **Idee:** ein kompakter Identitätsvektor reicht als einzige Bedingung für realistische Bilder | Code MIT; Gewichte auf SD-1.5-Basis (~0,9 Mrd. UNet-Parameter), Gewichtslizenz prüfen |
| [Vec2Face](https://github.com/HaiyuWu/Vec2Face) (ICLR 2025), geprüft | Feature-MAE + Decoder (ViT-Base) erzeugt Gesichter aus Identitätsvektoren; kleine Störungen des Vektors ergeben dieselbe Identität mit Variation; Attribute per Gradientenabstieg im Vektorraum | **Tool:** viel kleiner als Arc2Face, 112×112. **Idee:** Variation um den Code als Diagnose, was der Prior ergänzt | Code MIT; Gewichte auf HuggingFace, Lizenz prüfen |
| [EdgeFace](https://github.com/otroshi/edgeface) (TBIOM 2024), geprüft | Gesichtserkennung mit 1,24–18 Mio. Parametern; 1,77 M erreicht 99,73 % LFW | **Tool:** unabhängiger Identitätsprüfer, nur Evaluation; Vergleichsreferenz für den nativen Code | BSD-3 |
| [InsightFace/ArcFace](https://github.com/deepinsight/insightface), geprüft | Standard-Identitätsvektoren (512-d) | **Tool:** Prüfer, nur Evaluation; Vergleichsreferenz | Code MIT; vortrainierte Gewichte nur nichtkommerzielle Forschung |
| [MICA](https://github.com/Zielon/MICA) (ECCV 2022), geprüft | ArcFace-Merkmale → metrische 3D-Kopfform im FLAME-Modell | **Idee:** Identität als wenige interpretierbare Formparameter, getrennt von Ausdruck, Pose, Licht | Lizenz prüfen (FLAME hat eigene Lizenz) |
| [face-vid2vid](https://nvlabs.github.io/face-vid2vid/) (CVPR 2021), geprüft | Aussehen einmal aus einem Bild, danach pro Bild nur Keypoints; H.264-Qualität bei einem Zehntel der Bandbreite | **Idee:** Instanz einmal speichern, Veränderungen als kleine Deltas | NVIDIA-Lizenz prüfen |
| [Diffusion Autoencoders](https://diff-ae.github.io/) (CVPR 2022), geprüft | Zweiteiliger Code: semantischer 512-d-Code plus stochastischer Code für Rauschdetails; nahezu exakte Rekonstruktion mit beiden | **Idee:** zweiteiliger Code; ein aus dem Bild kodierter stochastischer Teil kann selbst beobachtete Details tragen, erst neu gesampelte Anteile sind ergänzt | Lizenz prüfen |
| StyleGAN2 + e4e/pSp-Inversion, Vorwissen | Bild → W/W+-Code eines Gesichtsgenerators (~30 Mio. Parameter) | **Idee/Tool:** kleiner realistischer Prior mit editierbarem Code | StyleGAN: NVIDIA nichtkommerziell; prüfen |

**Ausgeschlossen durch das Ziel:** Methoden, die pro Person Gewichte anpassen (Pivotal
Tuning, DreamBooth, LoRA pro Identität). Sie verletzen „Wissen zur Laufzeit ohne
Nachtrainieren“.

## Architekturideen, die über Gesichter hinaus tragen

1. **Identität getrennt von Störfaktoren.** Identität, Ausdruck, Pose, Licht und
   Hintergrund getrennt kodieren (FLAME/MICA, face-vid2vid, Arc2Face-Adapter). Nur
   der Identitätsteil gehört in den Instanzspeicher; der Rest ist Beobachtungszustand.
2. **Zweiteiliger Code: erinnert vs. ergänzt.** Wie bei Diffusion Autoencoders:
   ein gespeicherter Code plus frei gesampelter Rest. Mehrere Samples zeigen im
   Debug-Modus, welche Details variieren und also vom Prior stammen (vgl.
   Vec2Face-Störungen). Stabile Details sind nicht automatisch erinnert; auch der
   Prior kann sie stabil liefern.
3. **Prüfer als Maß statt Pixel-MSE.** Identitätstreue mit einem unabhängigen,
   eingefrorenen Erkenner messen (Verifikation gegen echte Bilder und ähnliche
   Distraktoren); Realismus getrennt. Der Prüfer darf nie Identitäten in den Graphen
   schreiben (kein verstecktes Oracle).
4. **Laufzeitverfeinerung durch Aggregation.** Mehrere Beobachtungen zu einem Code
   zusammenführen (in der Gesichtserkennung übliche Template-Mittelung); Qualität
   oder Unsicherheit pro Beobachtung gewichtet. Korrektur = Code revidieren und
   abhängige Zustände invalidieren, wie im [Ziel](integrated-latent-agent-goal.md).
5. **Deltas statt Neuspeichern.** Instanz einmal, Veränderungen (Pose, Ausdruck,
   Alter) als kleine Deltas (face-vid2vid).
6. **Einmal-Konditionierung genügt.** Arc2Face zeigt, dass ein einziger kompakter
   Vektor einen großen generativen Prior steuern kann. Für PATH-WM wäre die offene
   Frage, ob der eigene Instanzcode diese Rolle direkt übernehmen kann (nativ) oder
   ein kleiner Übersetzer zu einem externen Decoder nötig ist (zulässig laut Ziel).
7. **Code als Merkmal, nicht als Schlüssel.** Der Identitätsvektor ist nie Hash
   oder ID der Entität, sondern eine versionierte Komponente; Ähnlichkeitssuche
   liefert nur Kandidaten ([Entity-Memory-Design](entity-memory-design.md)).
   Stabiler wird er durch Invarianztraining, gelernte Unsicherheit pro Beobachtung
   (Probabilistic Face Embeddings, Shi & Jain 2019; Data Uncertainty Learning, 2020),
   unsicherheitsgewichtetes Zusammenführen und mehrere Merkmale. Fuzzy Extractors
   (Dodis et al. 2004) erzeugen exakte Schlüssel aus verrauschten Vektoren, trennen
   aber nicht besser als der Code und sind hier unnötig. Quellen aus Vorwissen.

## Übertragbarkeit über Gesichter hinaus (Vorwissen, nicht neu geprüft)

Das Muster „geteilter Prior in den Gewichten + kompakter Instanzcode + Decoder“ ist
allgemein. Gesichter sind nur der am besten ausgebaute Fall, weil es Millionen
beschrifteter Identitäten und eine eng ausgerichtete Domäne gibt.

| Domäne | Instanzcode | Generierung aus Code |
| --- | --- | --- |
| Stimme | Sprecher-Embedding (x-vector, ECAPA, ~192-d) | TTS konditioniert auf Sprecher-Embedding (Zero-Shot-Voice-Cloning) |
| Personen, Fahrzeuge | Re-Identifikations-Embeddings | selten; Körper über SMPL-Parameter |
| Tiere | SMAL-Formparameter | Rendering des 3D-Modells |
| Objekte allgemein | DINOv2-/CLIP-Bildmerkmale | IP-Adapter, BLIP-Diffusion, ELITE: Subjekt ohne Feintuning |
| 3D-Formen | DeepSDF/CodeNeRF-Codes (Form + Aussehen getrennt) | geteilter Decoder |

**Körperform gelernt statt SMPL (27. September, an Quellen geprüft):**
[RAC](https://github.com/gengshan-y/rac) (CVPR 2023) lernt aus monokularen Videos
ein Kategoriemodell (Menschen, Katzen, Hunde) mit Instanzcode für Morphologie,
Skelettmaße und Textur, getrennt von zeitlicher Artikulation und Verformung, ohne
3D-Scans. [3DInvarReID](https://arxiv.org/abs/2308.10658) lernt kleidungs- und
posenunabhängige Körperform-Merkmale für Personen-ReID. Beide passen zum Muster
„geteilter Prior + Instanzcode + Zustand pro Zeitpunkt“. Anders als bei Gesichtern
gibt es kein Arc2Face-Gegenstück mit riesigem Identitätsdatensatz: Kleidung
verdeckt die Form, und gewöhnliche ReID-Merkmale erfassen überwiegend Kleidung.

**Kleidung als eigene Schicht (an Quellen geprüft):** [CAPE](https://github.com/qianlim/CAPE)
(CVPR 2020) modelliert Kleidung generativ als Verschiebungen auf SMPL-Vertices,
abhängig von Kleidungstyp und Pose. [SMPLicit](https://enriccorona.github.io/smplicit/)
(CVPR 2021) beschreibt Kleidung als implizite Fläche mit latentem Code (Typ, Größe,
Weite, mehrere Lagen) auf dem SMPL-Körper. [BUFF](https://mlanthology.org/cvpr/2017/zhang2017cvpr-detailed/)
(CVPR 2017) schätzt die Körperform unter Kleidung aus Scan-Sequenzen. Alle hängen
an SMPL (eigene Lizenz, externe Abhängigkeit). Übertragbar ist die Idee ohne SMPL:
Kleidung als äußere Hülle, deren Beobachtungen die Körperform nach oben begrenzen;
mehrere Posen und Outfits engen sie weiter ein. Das ergäbe drei Lebensdauern:
Körpercode (langsam), Kleidungscode (pro Tag/Episode), Pose (pro Bild).

**Gelernte implizite Körpermodelle (SDF, an Quellen geprüft):**
[NPMs](https://github.com/pablopalafox/npms) (ICCV 2021) lernen aus deformierenden
Formen einen SDF-Formcode und einen Posecode (Fluss aus der kanonischen Pose),
ohne kategoriespezifische Handarbeit; neue Beobachtungen werden durch Optimieren
der Codes angepasst (Auto-Decoder). [imGHUM](https://arxiv.org/abs/2108.10842)
(ICCV 2021) ist ein generatives SDF-Modell des ganzen Menschen mit Form, Pose,
Händen, Gesicht und Korrespondenz-Semantik. [gDNA](https://arxiv.org/abs/2201.04123)
(CVPR 2022) erzeugt bekleidete Menschen als kanonische Form plus Normalendetails,
animiert über gelerntes Skinning ([SNARF](https://xuchen-ethz.github.io/snarf/)).
Alle lernen aus 3D-Scans oder 3D-Modelldaten (gDNA nutzt zusätzlich SMPL); RAC
kommt dagegen mit Videos aus. Für PATH-WM bräuchte ein SDF-Code 3D-Aufsicht oder
differenzierbares Rendern; vgl. den Geometrie-Cache in der
[Diskussion menschlicher Wahrnehmung](human-perception-discussion.md).

**Allgemeines Trainingsrezept:**
1. Identitätsencoder mit kontrastivem oder Margin-Verlust über Instanzen unter
   wechselnden Störfaktoren trainieren.
2. Decoder aus dem Code eine **andere** Ansicht derselben Instanz rekonstruieren
   lassen, mit Pose/Licht/Ausdruck als getrennter Eingabe. Das begünstigt, dass der
   Code die Identität trägt statt der Störfaktoren; die Trennung ist unabhängig zu prüfen.
3. Zur Laufzeit kodieren, mehrere Beobachtungen aggregieren, speichern.

**Woher Identitätslabels ohne Gesichtsdatensatz kommen:** synthetische Szenen mit
bekannten Identitäten (PATH-WM hat sie schon), Objektverfolgung in Video (dasselbe
Objekt über die Zeit als fehlbare Aufsicht) und Mehrfachansichten.

**Laufzeitlernen ohne Gewichtsänderung:** DeepSDF optimiert als „Auto-Decoder“ für
eine neue Instanz nur deren Code bei eingefrorenem Decoder. Das ist eine zweite
Variante zu Idee 4: Code per Gradientenabstieg statt per Encoder bestimmen.
Prototypical Networks bilden neue Kategorien aus wenigen Beispielen als Mittel der
Codes; das berührt das [Konzeptlernen](latent-concept-learning-review.md).

**Grenzen:** Deformierbare oder sehr variable Dinge (Kleidung, Szenen), Kategorien
ohne klare Identität, und Details, die nie beobachtet wurden. Ein neuer Instanzcode
setzt voraus, dass der Prior die Kategorie schon kennt; eine völlig neue Kategorie
erfordert Konzeptlernen, nicht nur Instanzspeicher.

## Tool oder nativ?

- **Nativ zuerst** gemäß der Regel [minimale externe Abhängigkeiten](experiment-workflow.md#keep-it-understandable).
  Externe Modelle nur als optionale, gekennzeichnete Prüfer (Identitätsverifikation)
  oder Debug-Renderer. Ein externer Decoder beweist nicht, dass der Agent die
  Identität selbst gelernt hat. Synthetische Instanzen mit bekannter Identität
  brauchen für erste Tests keinen externen Prüfer.
- **Nativ** ist das eigentliche Ziel: der Agent bildet den Instanzcode mit seinen
  eigenen Modulen. Ein Übersetzer vom nativen Code zu Arc2Face/Vec2Face wäre ein
  guter Test, ob der native Code die Identität enthält.
- Gesichtsspezifische Werkzeuge decken nur Gesichter ab; Ideen 1–5 sind allgemein.

## Bezug zum aktuellen Stand

Das [visuelle Gedächtnis](real-visual-memory-plan.md) zeigt Identitätserhalt bei
synthetischen Objekten, ohne realistische Ausgabe. Der jetzige Decoder ist auf
Treue (MSE) trainiert. Realistische Details wären Sache des im
[Bildcode-Plan](image-code-contract-plan.md#model-controlled-diffusion-transformer-with-direct-bypass)
vorgesehenen modellgesteuerten DiT, nicht eines neuen eigenständigen Decoders
(dort abgelehnt); siehe auch die
Photorealismus-Notiz in der [World-Labs-Sichtung](worldlabs-review.md).
Gesichtsdatensätze haben eigene Lizenz- und Einwilligungsbedingungen; vor einer
Verwendung prüfen.
