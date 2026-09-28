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
