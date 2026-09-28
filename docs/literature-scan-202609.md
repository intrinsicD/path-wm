# Literatur- und Newsscan September 2026 (Relevanz für PATH-WM)

28. September 2026, auf Alex' Wunsch. Quellen: letzte 15 Videos von YouTube @theAIsearch
(RSS-Feed, 23.08.–27.09.), Websuche zu World Models, kleinen multimodalen Netzen, ICL und
Laufzeit-Gedächtnis. Scholar-Inbox-Digest **nicht** enthalten (Anmeldung nötig, von Alex
selbst). 43 Kandidaten wurden per Hilfsagent zusammengefasst (2 Seiten nicht lesbar). Bewertung
unabhängig durch Fable 5.1 (`--effort max`) und Codex `gpt-6-astra` (xhigh) mit öffentlichem
Brief ohne Code/Messwerte; Belege und Volltexte: `runs/reviews/literature_scan_20260928/`.
Einschätzungen sind Hypothesen/Präferenzen der Reviewer, keine übernommenen Entscheidungen.

## Wichtigste Befunde (beide übereinstimmend)

1. **ICL und Aufgabenvielfalt erklärt unseren Transfer-Fehlschlag.** Unter einer Schwelle der
   Anzahl verschiedener Trainingsaufgaben verhält sich ein Transformer wie ein Bayes-Selektor über
   die trainierten Aufgaben und überträgt nicht auf neue (Raventós et al. 2023; Kirsch et al. 2022;
   Nguyen & Reddy „Differential learning kinetics“; Goddard et al., ICML 2025). „Wählt eine
   trainierte Regel statt der neuen“ ist genau diese Signatur. Kirsch et al. beschreiben zusätzlich
   ein Regime „viele Aufgaben, zu kleines Modell → lernt gar nicht“ – passend zu C3/L44.
   Einstiegsregel (Zipf-Kopf, Chan et al. 2022) hilft beim Entstehen, bringt aber keine Vielfalt.
2. **Kleinster entscheidender Versuch:** Anzahl verschiedener Trainingsregeln innerhalb einer
   Familie variieren, gleiche zurückgehaltene Regel(n), gleiche Architektur/Budget, 3 Seeds.
   Fable: generative Regelgrammatik bis |R| ≈ 10³–10⁴ mit Breiten-Arm 128/256; Codex: zuerst
   3 gegen ≈15 Regeln, Identifizierbarkeit prüfen, Held-out-Queries wählen, in denen die neue
   Regel allen trainierten widerspricht, und einen symbolischen Enumerator als Referenz.
   Unterscheiden: neue Parameterwerte (gleiche Familie) vs. neue Kompositionen vs. neue Primitive.
3. **Absicherung durch explizite Hypothesen:** Falls latente Induktion bei ≈0,6M Parametern auch
   mit viel Vielfalt nicht überträgt, ein expliziter Hypothesenspeicher mit Prüfer
   (Code-World-Model-Idee, DreamCoder, Programm-Enumeration) als Laufzeitpfad bzw. Baseline.
4. **Warnung zu E12 (Fable):** Der stärkste bekannte Hebel für Regelinduktion (ARC:
   Test-Time-Training, auch HRM-Nachanalyse) sind Gewichtsänderungen zur Testzeit – bei uns zur
   Laufzeit ausgeschlossen. Vorschlag: kleine, schnelle Mikro-Konsolidierungen prüfen.

## Weitere relevante Einträge

| Eintrag | Nutzen für uns |
| --- | --- |
| Latent Particle World Models (ICLR 2026 oral, Code) | objektzentrierte stochastische Dynamik mit latenten Aktionen; Baseline/Frontend für Pixel und Video, 8-GB-Trainierbarkeit prüfen |
| Factored Latent Action WMs; COMET | Latente Aktion je Entität, objekt-kausale Attention, latente MCTS: Vergleichsbaselines für Slots und Imagination |
| Low-rank latent carriers (192-d Modell) | billige Probe: liegt eine erschlossene Regel in einem niederrangigen Unterraum des Arbeitsbereichs? |
| MMLA; Position „Modular Memory“ | Rahmen für getrennte Lese-/Schreibdomänen, passt zu E11/E12; einfache episodische Abruf-Baseline zuerst |
| GPT-Policy | Protokollvorbild für Anpassung ohne Gewichtsänderung; kein Beleg für kleine Kerne |
| Dream RSI | Idee für Offline-Konsolidierung: gespeicherte exakte Evidenz als Simulator |
| Qwen3.5-0.8B / Needle 3 | Kontrollbaseline: dieselben Episoden als Text; überträgt ein vortrainiertes 0,8B-Modell? |

**Genannte Lücken der Liste:** Chan 2022, Raventós 2023, Kirsch 2022, Singh 2023 (transientes
ICL), Olsson 2022, PFNs/TabPFN, Neural Processes, MLC (Lake & Baroni 2023), PGM/ARC/Alchemy,
DreamCoder, Complementary Learning Systems, Slot Attention/SAVi/DINOSAUR/C-SWM, Neural
Production Systems, DreamerV3/TD-MPC2/V-JEPA, FSQ, BOCPD/Bayes-Surprise, HMR2.0/4DHumans.

## Für uns nicht relevant (8 GB, kleiner Kern)

Pixel-Video-„World Models“ (WorldCrafter, Lingbot World 2, Evoke, GWM Worlds 2, SolarWM,
VideoDeltaNet; 14B-Klasse, keine Zustandsdynamik im unseren Sinn), große Modelle/Agenten (CLM,
Limite 1B, Occamy, Isaac, UnifoLM, LLaDA image, Edge0, Bonsai 2 – 5,9 GB sind komprimierte Gewichte,
kein Trainingsbudget). 3D/4D-Werkzeuge (World Sculpt, One Video One World, 4DAnyone, Block 3D,
Orbit, UMR) erst als spätere Offline-Datenpipeline; VoiceMem/Audio8 TTS erst mit Sprache.

## Nachtrag: Scholar-Inbox-Digest und Websuche 15.08.–28.09. (28. September 2026)

Quellen: Scholar-Inbox-Digest 01.–28.09. (363 empfohlene Paper, fast alle 3D-Gaussian-Splatting/
Geometrie; 31 thematisch gefiltert), Websuche nach Paper vom 15.08.–28.09. (34 auf arXiv geprüfte
Einträge zu World Models, kleinen multimodalen Netzen, Gedächtnis, Denken in Schleifen,
Schichttypen, Text-Diffusion/Neurosymbolik). @theAIsearch: kein neues Video seit dem 27.09. Bewertet
haben unabhängig Fable 5.1 und Codex `gpt-6-astra` (xhigh), beide mit demselben öffentlichen Brief.
Belege: `runs/reviews/literature_scan_20260928b/`. Das oben genannte Verzeichnis
`runs/reviews/literature_scan_20260928/` des ersten Scans existiert nicht; dessen Belege fehlen.
Alles unten sind Prüfvorschläge, keine übernommenen Entscheidungen.

### Beide Reviewer übereinstimmend

| Eintrag | Problem | Kleinster Check |
| --- | --- | --- |
| Readout Feedback (RoFB, 2608.24136) | Denkschleifen ohne Gewinn | eingefrorener Checkpoint, 1/2/4/8 Schleifen mit/ohne Rückführung der Zwischenvorhersage; Codex: erst prüfen, ob ein vorhandener Rückkanal reicht, sonst ist es kein trainingsfreier Test |
| MixerLoop (2608.18230) | Schleifen, Effizienz | nur Attention wiederholen, FF einmal, bei gleichem Rechenaufwand; relativiert das Wiederholen des ganzen Blocks, widerlegt den geteilten Kern nicht |
| GeoCo-SAVi (2609.06628, Code) | Form-/Größenerhalt der Slots | Farbe/Textur bei fester Geometrie variieren, Größen-/Positionsdrift messen, dann ein Alignment-Verlust |
| LEON (2608.27259) | Vorhersage schlägt Kopie nicht | operatorbasierter Übergang gegen geteilten Kern, Kopie und konstante Geschwindigkeit bei eingefrorenem Encoder/Decoder |
| Narcissus (2608.25657) | Regeltransfer | kleine Regelgrammatik + Verifier auf dem E4-Split als Obergrenze bzw. Absicherung per expliziten Hypothesen |
| Spectral-Target JEPA (2609.04264) | Latents ohne Physik | lineare Probe auf Verschiebung/Geschwindigkeit; Fourier-Hilfskopf nur beim Training |
| Displacement Geometry (Digest 2609.24209) | Encoder-Adaption, Konzepte | latente Differenzvektoren (nur Form bzw. nur Größe geändert) vor/nach Adaption bzw. Konsolidierung vergleichen |

Nur Fable in der Spitzengruppe: RecurTrace (2609.03379; Loop-Memory-Attention, Ablation bei fester
Schleifenzahl), „Better Slots, Better Worlds“ (2608.12078; Slot-Bindungsmetriken mit
Vorhersagefehler korrelieren). Nur Codex: FuseReg (Digest; Decoder auf zufälligen Teilmengen von
Encoderstufen zeigt, wo Detail verloren geht), MO-IKE/„Beyond Endpoint Scores“ (Korrektur bei
eingefrorenen Gewichten auf Zuverlässigkeit, Generalisierung, Spezifität prüfen; Verläufe statt
Endwerte).

**Schwächt aktuelle Annahmen (keine widerlegt):** Kollapsfreie oder rekonstruierbare Latents
tragen nicht automatisch Physik/Geometrie (Spectral-Target, GeoLAM); vortrainierte Features tragen
die Robustheit objektzentrierter Weltmodelle (Better Slots) – Risiko für von Grund auf trainierte
schlanke Encoder; Feintuning kann relationale Geometrie zerstören (Warnung für Offline-Konsolidierung).

**Nur lesen:** ForeWAM, Gated Recurrent Transformers, T-LoopFormer, MemBodied, Info3R, SURE-Map,
ShapeLex, Continual-WM-Benchmark. **Nicht relevant:** Roboter-WAMs/DiTs mit großen Backbones
(SlotDiT, PointCast, Rolling-/DualWAM, SG-/StageWAM, DyMD, LeFlow), 3D-Rekonstruktion, große
Diffusions-/Text-Diffusionsmodelle (RMDM, PlaidQ – zurückgezogen, IIF), MoE-/LLM-Skala-Arbeiten,
VLM-Vortraining (Semantic Serialization, MMCS), EBM-AE (MNIST).

**Lücken:** Im Fenster keine neuen, prüfbaren Arbeiten zu EBMs, Flow-LMs oder ICL-Theorie und kaum
kleine multimodale Netze mit echt geteilter Repräsentation. Aus eigenem Wissen: TRM (2510.04871,
von Codex geprüft), HRM, Huginn/Mixture-of-Recursions, Coconut, Titans, DINO-WM/V-JEPA 2,
Energy-Based Transformers (Fable, ungeprüft). Exakte Attention unter unseren Masken/Halbpräzision
bleibt lokal zu messen.

### Nachtrag 2: Contrastive World Models, Scholar-Inbox-Trending, August-Digest

Auf Alex' Hinweis: [Contrastive World Models](https://arxiv.org/abs/2609.22175) (Li, 27.08.) fehlte
im Digest, weil Alex' Scholar-Inbox-Profil es mit −59 bewertet; World-Model-Paper werden dort
systematisch herausgefiltert. Trending (8 Einträge, aus von Alex gespeicherter Seite) und
August-Digest 15.–31.08. (286 Paper; Fenster 15.–19.08. bei 100 gekappt) ergänzt. Gegenbefund:
Ramakrishnan et al. 2023 ([2401.00057](https://arxiv.org/abs/2401.00057)): objektzentrierte
kontrastive Weltmodelle zerfallen bei neuen Attributen/Kombinationen. Beide Reviewer erneut
unabhängig; Belege im selben Verzeichnis (`*_followup.md`).

**Contrastive World Models (beide):** hohe Priorität als Kontrolle/Diagnose für „Vorhersage schlägt
Kopie nicht“, Übernahme nur bedingt. Pixelverluste werden von statischem Inhalt dominiert, so dass
Kopieren fast optimal ist. Kein Ersatz des Decoders; zuerst:
1. Vorhandenen Checkpoint gegen Kopie und konstante Geschwindigkeit getrennt auf bewegten/unbewegten
   Bereichen, Positionen und Identität auswerten (Codex: auch kurze freie Rollouts).
2. Erst dann zwei gleich initialisierte kurze Läufe: Pixelverlust gegen Pixelverlust plus kleinen
   kontrastiven Zukunftsterm auf lokalen Encoder-Exporten.
3. Auswertung mit zurückgehaltenen Farb-/Form-Kombinationen und Hintergrundwechsel (wegen des
   Gegenbefunds); sinkender Kontrastivverlust allein ist kein Erfolg.
Risiken bei 8 GB: wenige Negative bei kleinen Batches, fast identische Negative in statischen
Szenen, Halbpräzision und Logit-Skalierung, Verlust von Form/Größe als „irrelevanter“ Information.

**Weitere:** RigidBench (2608.15555): Bildähnlichkeit (SSIM) korreliert nicht mit
Trajektorienfehler – Bewegung, Geometrie und Identität getrennt messen (Codex hohe Priorität,
Fable Warnung). Read-Write-Relax (2608.21677): Attention auf wenige latente Tokens wirkt als
Tiefpass; Fable schlägt einen Spektralvergleich des Vorhersagefehlers gegen die Kopie vor.
Floorplan-Readout (2608.25608): Fusion statt Konditionierung zweier Ausgaben, Vorsicht bei
Readout-Rückführung. WTF?!, übrige Trending-Einträge und restliche August-Kandidaten: nicht relevant.

**Top-Liste danach:** Fable behält seine Reihenfolge und setzt Contrastive World Models neben
Spectral-Target (Schritt 1 vor dem Fourier-Kopf). Codex: Narcissus, GeoCo-SAVi, RigidBench-Kontrollen,
Contrastive World Models vorn; MixerLoop und MO-IKE vorläufig heraus. Keine zentrale Entscheidung widerlegt.
