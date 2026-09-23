# Gemeinsame Repräsentation: implementierbarer Prüfentwurf

23. September 2026. Status: mathematischer Versuchsplan, keine implementierte
Modellfunktion und kein Trainingsergebnis. Auf Alex' Wunsch wurde der Entwurf mit
Claude Opus 5.5 bei explizitem Aufwand `max` kritisch abgeglichen. Abschließende
Reviewbelege und Korrekturen stehen am Ende. Der Plan konkretisiert den
[Prüfvorschlag](latent-concept-learning-review.md#prüfvorschlag-gemeinsame-repräsentation-über-abstraktionsarten)
und dient dem [integrierten Forschungsziel](integrated-latent-agent-goal.md).

## 1. Hypothese und Aussagegrenze

Geprüft wird, ob ein festes Format und geteilte gelernte Operationen verschiedene
Arten von evidenzabhängigen Funktionen effizient erschließen und anwenden können:

\[
 Z_c=G_\theta(S_c),\qquad
 p_\theta(y=1\mid x,S_c)=\sigma(F_\theta(x,Z_c)).
\]

`c` bezeichnet einen Wissenseintrag, `S_c` seine gültigen Belege und `Z_c` einen
abgeleiteten Zustand. Alle Familien verwenden in Arm A dieselben Parameter θ.
Verschiedene Inhalte haben verschiedene Zustände. Kompatibilität bedeutet hier
funktionierende gemeinsame Operationen; gleiche Tensorform oder ähnliche
Embeddingabstände allein reichen nicht. Innere gemeinsame Semantik und beliebige
Abstraktionsarten werden durch diesen Versuch nicht bewiesen.

Die kleine endliche Grammatik ist absichtlich vollständig prüfbar. Ihre Funktionen
sind alle durch endlich viele Bits beschreibbar. Ihre abstrakte Darstellbarkeit
ist deshalb nicht die offene Frage; zu prüfen sind Lernen, Übertragung, Nutzung,
Korrektur und Kosten einer gewählten Architektur. Der binäre Ausgabevertrag
vereinheitlicht den Leser und begrenzt den Nachweis auf diese Aufgaben.

## 2. Datenvertrag und unabhängige Aufgabenlösung

Zwei geordnete Objekte besitzen je vier kategoriale Attribute:

\[
 a,b\in\{0,1,2,3\}^{4},\quad s,u\in\{0,1\},\qquad
 x=[\operatorname{onehot}(a),\operatorname{onehot}(b),s,u]\in\mathbb R^{34}.
\]

`a` und `b` sind unterschiedliche Rollen; Attributpositionen haben feste Bedeutung.
Die jeweils irrelevanten Attribute, `s` und `u` werden ebenfalls zufällig gezogen.
Es gibt keine Objekt-ID, Regellabels oder Familien-ID im Modellinput. `s` ist der
aktuelle binäre Zustand; `u` ein vom Generator gesetzter Eingriff. Alle Ausgaben
sind ein einzelnes Bit. Der Evaluator kennt die verborgene Regel h, das Modell nicht.

Drei disjunkte Funktionsfamilien:

\[
\begin{aligned}
 h^{cat}_{j,V}(x)&=\mathbf1[a_j\in V],
 &j\in\{0,\ldots,3\},\quad V\subset\{0,1,2,3\},\ |V|=2;\\
 m_{jk\delta}(a,b)&=\mathbf1[(a_j+\delta)\bmod4=b_k],
 &j,k\in\{0,\ldots,3\},\quad\delta\in\{0,\ldots,3\};\\
 h^{rel}_{jk\delta}(x)&=m_{jk\delta}(a,b);\\
 v&=u\,m_{jk\delta}(a,b);\\
 h^{tr}_{jk\delta,open}(x)&=s\lor v;\\
 h^{tr}_{jk\delta,close}(x)&=s\land\neg v;\\
 h^{tr}_{jk\delta,toggle}(x)&=s\oplus v.
\end{aligned}
\]

24 Kategorie-, 64 Relations- und 192 Übergangsregeln, insgesamt 280. Die Relation
ist ein geordneter Attributvergleich; der Erstversuch behauptet kein allgemeines
räumliches oder physikalisches Verständnis. Die Übergänge sind Wirkungen des
bekannten Simulators unter gesetztem `u`, keine aus Videos identifizierte Kausalität.

Alle 262144 Eingaben wurden für diesen Entwurf enumeriert: 280 verschiedene
Wahrheitsfunktionen. Gleichnamige oder gleichwirkende Regeln dürfen nie über einen
Train/Test-Split hinweg dupliziert werden. Der unabhängige Prüfer benötigt eine
zweite, direkt boolesche Auswertung zusätzlich zur vektorisierten Implementierung.

### Episoden und Stichproben

Eine Episode wählt Familie `f` gleichverteilt, dann `h` gleichverteilt innerhalb
ihres Split-Anteils. Support und Query stammen aus derselben Regel:

\[
 S=\{(x_i,h(x_i))\}_{i=1}^{N},\qquad
 Q=\{(x_j,h(x_j))\}_{j=1}^{M},\quad X_S\cap X_Q=\varnothing.
\]

Eingaben werden gleichverteilt ohne Zurücklegen gezogen. Support nicht anhand
verborgener Labels balancieren und nicht bis zur gewünschten Erkennbarkeit ziehen:
diese Selektion verändert die Evidenzverteilung. Queryeingaben stammen ebenfalls
uniform aus dem verbleibenden Definitionsbereich. Masken und Supportgröße sind
sichtbar; Padding enthält keine Information.

Bei unzureichendem Support kann mehr als eine Regel passen. Das ist keine
Modellfehlleistung allein. Deshalb werden Ambiguität und Leistung der expliziten
Referenzinferenz zusätzlich ausgewiesen; schwierige Episoden werden nicht heimlich
entfernt. Eine bekannte Frage/Rollenkodierung ist noch keine selbständige Entdeckung
des interessanten Abstraktionskriteriums.

## 3. Tensoren und gemeinsamer latenter Kern (Arm A)

| Name | Form | Bedeutung |
| --- | --- | --- |
| `support_x` | `[B,Nmax,34]` | Supporteingaben |
| `support_y` | `[B,Nmax,1]` | ausschließlich Supportlabels |
| `support_valid` | `[B,Nmax]` bool | gültige Belege |
| `query_x` | `[B,M,34]` | neue Eingaben |
| `query_y` | `[B,M,1]` | nur Loss/Evaluator |
| `Z` | `[B,K,d]` | komprimierter Wissenszustand |
| `logits` | `[B,M,1]` | unnormalisierte Antwort |

Vorschlagswerte: `d=64`, `K=8`, vier Attentionköpfe, `d_head=16`, `L=2`
Wiederholungen desselben Zustandsblocks, FF-Breite `4d`, GELU, LayerNorm ε=`1e-5`,
Dropout `0`, zunächst FP32. Budgetkurve `(K,d)∈{(2,8),(4,16),(8,64)}`, jeweils vier Attentionköpfe;
keine Annahme einer besonderen Bedeutung einzelner Slots. Standardlineare Initialisierung, Seedslots normal mit
Standardabweichung `0.02`. Der Vorschlag setzt keinen existierenden Modellcheckpoint
oder Decoder voraus.

### Attention, Masken und leere Evidenz

Für `Q∈R^{B×nq×d}`, `H∈R^{B×nk×d}` und Kopf `r`:

\[
 q_r=QW^Q_r+b^Q_r,\quad k_r=HW^K_r+b^K_r,\quad v_r=HW^V_r+b^V_r,
\]
\[
 A_r(Q,H;M)=\operatorname{softmax}_{keys}
 \left(\frac{q_r k_r^\top}{\sqrt{d_h}}+M\right)v_r,
\]
\[
 \operatorname{MHA}(Q,H;M)=\operatorname{concat}_r(A_r)W^O+b^O.
\]

`W^Q_r,W^K_r,W^V_r:[d,d_h]`, `W^O:[d,d]`; lineare Biases aktiviert.
`M:[B,1,nk]` wird über Köpfe und Queries gebroadcastet, gültig=`0`, ungültig=`−∞`.
Ein gelernter Nullbeleg ist in jeder Episode gültig. Damit entsteht auch bei `N=0`
kein Softmax über ausschließlich `−∞`. Sonstige Paddinginhalte werden durch Masken
vollständig ausgeschlossen. Keine Supportpositions- oder Zeitkodierung. Doppelte Beleg-IDs vorab entfernen.
Mehrfaches Kopieren einzelner inhaltlicher Beispiele verändert Attentiongewichte;
Permutationstreue bedeutet nicht Unempfindlichkeit gegen solche Wiederholungen.

\[
 E_i=\operatorname{MLP}_{35\to d\to d}([x_i,y_i]),\quad
 H=[e_{null},E_1,\ldots,E_{Nmax}],\quad Z^{(0)}=Z_{seed}.
\]

MLP besitzt GELU zwischen zwei linearen Schichten. Jede Verwendung unten hat eigene
Parameter, außer ausdrücklich geteilten Wiederholungen. `LN_Z` und `LN_H` sind
separate LayerNorms; die einzelnen Sublayer haben eigene Normalisierungen.

Für `ℓ=0,…,L−1`:

\[
\begin{aligned}
 U&=Z^{(\ell)}+\operatorname{MHA}_{cross}(\operatorname{LN}_1 Z^{(\ell)},
                                       \operatorname{LN}_H H;M_S),\\
 V&=U+\operatorname{MHA}_{self}(\operatorname{LN}_2 U,\operatorname{LN}_2 U;0),\\
 Z^{(\ell+1)}&=V+\operatorname{FF}_{d\to4d\to d}(\operatorname{LN}_3 V),\\
 Z&=\operatorname{LN}_{out} Z^{(L)}.
\end{aligned}
\]

Die Parameter dieser drei Sublayer sind über `ℓ` geteilt; Cross- und Self-Attention
haben getrennte Parameter. Kompression hängt ausschließlich von Support ab.
Sie wird einmal pro Episode berechnet, nicht erneut je Query.

### Gemeinsamer Leser

\[
\begin{aligned}
 q&=\operatorname{MLP}_{34\to d\to d}(x),\\
 r&=q+\operatorname{MHA}_{read}(\operatorname{LN}_4 q,\operatorname{LN}_5 Z;0),\\
 t&=r+\operatorname{FF}_{read}(\operatorname{LN}_6 r),\\
 \ell(x,Z)&=w^\top\operatorname{LN}_7(t)+b,\qquad p=\sigma(\ell).
\end{aligned}
\]

Queries dürfen als Batch verarbeitet werden, aber lesen einander nicht. Kein Zugriff
des Lesers auf Support, Regel-ID oder Querylabels. Es gibt einen Eingangspfad `x`
zum Leser: Er ist für Anwendung notwendig, ermöglicht jedoch Prior-/Kopierstrategien,
die durch die Kontrollarme explizit gemessen werden.

`K×d` ist eine begrenzte Schnittstelle, kein belegter Speichervorteil. Ein Eingabebeleg
benötigt dicht gepackt 19 Bits (acht Attribute zu je zwei Bits, s, u und y); die
Identität einer eindeutig bestimmten Regel braucht höchstens neun Bits; eine
Versionsmenge über die vorgegebene Grammatik lässt sich als 280-Bit-Maske darstellen. `K=8,d=64` in FP32 belegt 16384 Bits und
ist damit größer als 64 gepackte Belege (1216 Bits, jeweils ohne Metadaten).
Der Erstversuch prüft einen gelernten Leser mit wiederverwendbarem Zustand; eine
Kompressionsbehauptung benötigt eine eigene Qualitäts-/Bitratenkurve einschließlich
Belegspeicher und Quantisierung. Tokenzahl ist nicht Informationsgehalt.

## 4. Lernziel und Gradientengrenzen

Für binäre Labels stabile Logit-BCE. Der Hauptloss wird auf den balancierten
Erfolgsscore abgestimmt. Setze `t_h(x)=h(x)` für Kategorie/Relation und
`t_h(x)=h(x) XOR s` für Übergänge; `ρ_h=P_x(t_h(x)=1)` unter uniformen Eingaben:

\[
 \rho_{cat}=\tfrac12,\quad \rho_{rel}=\tfrac14,\quad
 \rho_{open}=\rho_{close}=\tfrac1{16},\quad\rho_{toggle}=\tfrac18,
\]
\[
 w_h(x)=\frac{t_h(x)}{2\rho_h}+\frac{1-t_h(x)}{2(1-\rho_h)},\qquad
 \mathbb E_x[w_h(x)]=1,
\]
\[
 bce(\ell,y)=\operatorname{softplus}(\ell)-y\ell,\qquad
 \mathcal L=\frac13\sum_f\frac{1}{|\mathcal B_f|}
 \sum_{e\in\mathcal B_f}\frac1M\sum_{j=1}^M w_{h_e}(x_{ej})bce(\ell_{ej},y_{ej}).
\]

`h`, `ρ` und Gewichte sind ausschließlich Generator-/Losswissen; sie gehen nicht
als Eingaben in G oder F ein. In PyTorch `binary_cross_entropy_with_logits`
mit `reduction="none"` berechnen, elementweise mit w multiplizieren und durch die
Queryanzahl mitteln. `pos_weight` ist hier falsch, weil es nach y statt nach t
gewichtet; ebenso nicht durch die zufällige Summe der Gewichte dividieren. Support bleibt ungewichtet und uniform. Der Sigmoidwert
`p` bezeichnet im Hauptarm einen kostenabhängigen Entscheidungsscore, keine als
kalibriert behauptete natürliche Wahrscheinlichkeit. Ungewichtete BCE bleibt eine
zusätzlich berichtete Fehlergröße. Eine spätere Wahrscheinlichkeitskalibrierung wäre
ein eigener, ungewichteter Lernvertrag.

Der Batch enthält gleich viele Episoden jeder Familie. Gradienten laufen von
gewichteter Query-BCE durch Leser, `Z`, Slotblock und Supportencoder. Kein `detach(Z)` im Training.
`support_y` sind Eingangsdaten, `query_y` ausschließlich Zielwerte. Keine frei
mittrainierten Ziel-Embeddings, kein ausschließliches Latent-MSE, keine Supervision
mit der verborgenen Regelbeschreibung. Gute BCE allein beweist keine Nutzung von Z.

Training verändert θ. Evaluation verwendet `model.eval()`, unveränderte Parameter
und Buffer sowie `no_grad()`. Support, Belegrevisionen und Z dürfen sich ändern.
Eine Optimierung von Z oder θ während Evaluation ist in diesem Versuch deaktiviert.
BatchNorm und testabhängige laufende Statistiken kommen nicht vor.

Vorschlagskonfiguration für eine spätere Recipe: AdamW, `lr=3e-4`,
`betas=(0.9,0.999)`, `weight_decay=0.01`, Gradientennormgrenze `1`, `B=12`, `M=32`,
`N=0` mit Wahrscheinlichkeit `0.05`, sonst uniform über die vorab gewählte
Zweierpotenz-Supportgrößenliste aus Abschnitt 8, 5000 Updates pro Seed und Arm. C_full und C_native aktualisieren jede
Familienkopie auf ihren vier Episoden mit eigenem Optimizer und mittlerem Familienloss;
A, A+ID, B und D verwenden einen Optimizer für den familiengemittelten Batchloss.
Hauptvergleich bei `K=8`,
fünf Seeds `{11,23,47,59,83}`, Seeds für Datenströme getrennt speichern. Weitere
Budgets und ein längeres Training wären neue vorab erklärte Vergleichsläufe.
Kein Training wird durch dieses Dokument gestartet; Laufzeit/GPU-Passung ungemessen.
Der letzte Checkpoint ist der feste Auswertungscheckpoint. Lernkurven werden
vollständig erhalten; kein nachträgliches Aussuchen des besten Testzeitpunkts.

## 5. Vergleichsarme und konkrete Spezialformate

**A:** oben definierter geteilter Encoder, Zustandsblock und Leser, keine Familien-ID.

**A+ID:** wie A; nur zum Queryembedding wird `E_family[f]∈R^d` addiert.
G erhält weiterhin keine Familien-ID. Dieser Pflichtdiagnosearm zeigt, wie viel
der Abstand zu gerouteten Lesern auf Familieninferenz entfällt.

**B:** derselbe gemeinsame G, aber drei unabhängige Kopien des Lesers F, geroutet
nach wahrer Familie. A+ID gegen B prüft gemeinsame gegenüber spezialisierten Lesern
bei gleicher verfügbarer Familieninformation, aber unterschiedlicher Parameterzahl.

**C_full:** drei unabhängig trainierte Paare `(G_f,F_f)` mit derselben Tensorform
wie A und Routing nach Familie. Dies ist die primäre Spezialistenreferenz. Rund
dreifache Gesamtparameter und pro-Familie-Lernverfahren sind ausdrücklich zusätzliche
Ressourcen. Gleiche Tensorform bedeutet nicht automatisch kompatible Semantik;
unabhängiges Training beweist umgekehrt keine prinzipiell verschiedene Geometrie.
C_full gegen B untersucht die zusätzliche Spezialisierung von G, wiederum unter
Offenlegung veränderter Parameterzahl. Eine parameterangepasste C-Variante folgt
gegebenenfalls als eigener Budgetpunkt.

**C_native, optional:** drei gelernte Support-zu-Parameter-Modelle mit strukturiertem Zustandsformat.
Der Supportencoder und Slotblock entsprechen C_full; statt allgemeinem Leser wird aus
`mean_slots(Z)` eine parametererzeugende MLP `64→256→n_f` mit GELU angewendet. Ihre Ausgabe
ist der einzige persistierte Wissenszustand:

\[
\begin{aligned}
 z_{cat}&=(w,\beta), & \ell_{cat}&=w^\top A+\beta, & A&=onehot(a)\in\mathbb R^{16};\\
 z_{rel}&=(W,\beta), & \ell_{rel}&=A^\top WB+\beta, & B&=onehot(b)\in\mathbb R^{16};\\
 z_{tr}&=(W_{su},\beta_{su})_{s,u\in\{0,1\}},
 &\ell_{tr}&=A^\top W_{su}B+\beta_{su}.
\end{aligned}
\]

Zustandsgrößen: 17, 257 und 1028 FP32-Werte; nicht künstlich als gleich groß ausgeben.
`W:[16,16]`, Übergangstensor `[2,2,16,16]` plus Bias `[2,2]`. Gemeinsame Eingaben
bleiben `[34]`; der native Leser verwendet die aufgeführten Teile. Der Endzustand
kann jede verborgene Boolesche Regel der jeweiligen Familie durch Logits beliebig
genau darstellen. Die gelernte Erschließung dieser Parameter bleibt zu prüfen.
C_native erhält die Familie durch Routing und die strukturelle Vorannahme durch den Leser.
Erfolg wäre keine faire Widerlegung eines parameter-/speicherärmeren A allein.

**D:** Supportencoder wie A, dann zwei geteilte Self-Attention-/FF-Schleifen über
`H`; Queryleser wie A liest die gültigen Belegtokens direkt. Nullbeleg und Masken
wie oben. Kein fester K-Slot-Flaschenhals; gespeicherte Belege bzw. vorbereitete
Tokens, Vorbereitungskosten und Kosten jedes Queryzugriffs separat zählen.
Dies ist ein leistungsfähiger vollständiger Belegleser, kein schwacher
Nearest-Neighbor-Vergleich. Beschränkter Abruf wäre ein späterer eigener Faktor.

**O:** endliche Hypotheseninferenz als Evaluationsdiagnose, nicht gelernter Konkurrent.
Sie kennt Grammatik und deklarierten Episodenprior; keine ihrer Zustände oder
Antworten dürfen in A–D eingespeist werden.

Gleiche Episoden-IDs und Daten pro Arm. B/C_full/C_native bekommen pro Familie dieselbe Anzahl
Support-/Querybeispiele wie A, keine zusätzlichen Updates mit zusätzlichen Daten.
Logisch gepackte Belege, tatsächlich gespeicherte Belege und vorbereitete Tokens
getrennt zählen. Gesamtparameter, aktive Parameter, gespeicherte Bytes, temporäre Aktivierungen,
Evidenzzugriffe, Trainings-/Inferenz-FLOPs und Latenz getrennt protokollieren.
Format, Reader und Vorannahmen ändern sich in C gemeinsam: C_native bewertet ein Paket,
keinen isolierten kausalen Effekt der Geometrie. Die neuronale Zuordnung von
Support zu Operatorparametern wird gelernt; das native Format ist kein exaktes
Regel- oder Posteriororacle. Auch A gegen B ist durch Parameterzahl und Routing konfundiert; die Diagnose muss dies benennen.

Eine optionale kombinierte Abfrage verwendet zwei getrennt erschlossene Zustände:
`p1=F(x,Z1)`, `p2=F(x,Z2)`. Beide Einzelantworten und anschließend ihre Konjunktion
werden geprüft. Eine fest programmierte Konjunktion belegt gemeinsame Verwendung,
keine gelernte Kompositionsoperation. Diese Erweiterung ist kein Bestehensgate des
Erstversuchs. Geteilte Gewichte schließen intern gelernte Spezialisierung nicht aus;
deren Abwesenheit oder eine universelle gemeinsame Geometrie wird nicht behauptet.

## 6. Splits, Unbestimmtheit und Referenzposterior

Regeln kanonisch als JSON ohne Leerzeichen serialisieren und mit SHA256 über
`"shared-abstraction-v1" + json_key` sortieren; Vergleich des Hashhexstrings aufsteigend.
Die konkrete [Gruppenliste](../runs/reviews/shared_abstraction_formulas_20260923/split-proposal.json)
fixiert den Vorschlag.
Kategorien werden mit ihrem Komplement gruppiert: Schlüssel `(j,min(V,V^c))`;
12 Gruppen, davon 8 Train, 2 Validation, 2 Test. Alle drei Übergangsarten und die
Relation mit gleichem `(j,k,δ)` gehören in dieselbe Gruppe: 64 Gruppen, davon
44 Train, 10 Validation, 10 Test. So gelangt dieselbe Matchingregel nicht über eine
andere Familie in den Testsupport. Gruppenzuteilung als explizite JSON-Liste speichern.
Jede Attributposition, jeder δ-Wert und jede Übergangsart muss im Training vorkommen;
der Manifestvalidator lehnt andernfalls den Split vor jedem Modelllauf ab.

Das ergibt Train/Validation/Test-Regelzahlen:
Kategorien `16/4/4`, Relationen `44/10/10`, Übergänge `132/30/30`.
Dies prüft ungesehene Regeln bzw. Faktorkombinationen in einer vorgegebenen Grammatik.
Nur vier Kategorieregeln im Test begrenzen die Aussage ausdrücklich. Die von Claude
vorgeschlagenen vier komplement-/komponentengeschlossenen Folds würden jede Regel
einmal zurückhalten, erfordern aber eine eigene vierfache Trainingsplanung.
Zusätzlicher Test mit neuen Eingaben bekannter Trainingsregeln trennt Instanztransfer.
Ein separater Leave-one-family-out-Lauf trainiert ohne eine Familie; B/C_full/C_native und A+ID besitzen
dann für diese Familie keinen trainierten Leser beziehungsweise kein trainiertes Familienembedding und sind dort keine Vergleichsarme.
A und D sowie der Oracle dienen dieser gesonderten Prüfung.

Bei deklarierter Hypothesenmenge `H_eval` und Prior π:

\[
 \pi_S(h)=\frac{\pi(h)\prod_i\mathbf1[h(x_i)=y_i]}
                  {\sum_g\pi(g)\prod_i\mathbf1[g(x_i)=y_i]},\qquad
 p_O(y=1\mid x,S)=\sum_h\pi_S(h)h(x).
\]

Primäre Informationsreferenz `O_full` verwendet alle 280 Regeln mit
familienuniformem Prior. Sie vermeidet Splitwissen, ist dadurch aber kein exakter
Bayes-Prädiktor des engeren Testgenerators. `O_fam` kennt zusätzlich die Familie.
Beide bleiben getrennt ausgewiesen. Im zusätzlich ausgewiesenen genauen Splitprior-Oracle ist `H_eval` die zulässige Regelmenge des jeweiligen
Evaluationsarms und `π(h)=1/(3|H_{eval,f}|)` bei drei gleich wahrscheinlichen Familien.
Es kennt den Splitprior; dies ist privilegierte Evaluatorinformation. Es darf nicht als Informationsstand der neuronalen Modelle ausgegeben werden. Ein familienkundiges Oracle
konditioniert π zusätzlich auf f; immer getrennt vom familienblinden Oracle berichten.

Nenner `0` ist im rauschfreien Versuch ein Daten-/Versionsfehler, kein Fall für
stilles uniformes Zurücksetzen. Queryantwort-Eindeutigkeit kann auch bei mehreren
verbleibenden h gegeben sein. Posteriorentropie und Querydisagreement sind getrennte
Diagnosen. Der Oracle definiert eine Inferenzreferenz für natürliche Queryverteilung;
er ist nicht automatisch die optimale Entscheidung unter jedem regewichteten Score.
Für den balancierten Hauptscore verwendet die Oracleentscheidung dieselben Gewichte:

\[
 U_c(x,S)=\sum_h\pi_S(h)w_h(x)\mathbf1[h(x)=c],\qquad
 \widehat y_O=\mathbf1[U_1\ge U_0],\quad q_O=\frac{U_1}{U_0+U_1}.
\]

Die U-Regel optimiert den populationsgewichteten Mittelwert unter ihrem deklarierten
Prior, nicht separat jede Familie oder die Wahrscheinlichkeit, dass alle Gates
bestehen. Empirische Quotienten-Scores sind endlichstichprobenabhängig.
Unter dem jeweiligen deklarierten Prior bleibt `p_O` die Ambiguitätsdiagnose; `q_O` ist der
kostenabhängige Entscheidungsscore. Gewichte beziehen sich auf den gesamten
uniformen Definitionsbereich; das Ausschließen von Supporteingaben erzeugt eine
kleine endliche Abweichung. Eine exakte Restmengenvariante ersetzt `ρ_h` durch
`(D ρ_h − Anzahl positiver t_h im Support)/(D−N)`, `D=262144`, konsistent in Loss
und Oracle. Nicht beide Varianten innerhalb eines Laufs mischen. Der Startentwurf
verwendet die feste Domänengewichtung; bei einem später eingeschränkten Eingaberaum
(z. B. zurückgehaltenen Objekten) müssen ρ und Referenzfloors dort neu enumeriert werden; die empirische Balanced Accuracy wird stets
aus ihren beiden tatsächlichen Klassenanteilen berechnet.

## 7. Runtime-Korrektur und Persistenz

Autoritative Belege werden als `e=(concept_id,evidence_id,revision,x,y,active)`
gespeichert; IDs/Revisionen sind Metadaten außerhalb des Modells.

\[
 S_c^{(r)}=\operatorname{ActiveEvidence}(store,c,r),\qquad
 Z_c^{(r)}=G_\theta(S_c^{(r)}).
\]

`append` fügt einen neuen korrekten Beleg hinzu. `replace(id,x,y)` deaktiviert die
alte Version und erzeugt eine neue. `retract(id)` deaktiviert sie. Danach wird Z
vollständig aus gültigen Belegen neu berechnet. Das ist eine referenzierbare
Laufzeitaktualisierung ohne Gewichtstraining, kein gelernter inkrementeller Updateoperator.
Kosten wachsen mit Supportumfang. Der komprimierte Zustand ersetzt den Belegspeicher
in diesem Versuchsarm nicht; Code-only-Anwendung und gesamte persistierte Bytes
müssen getrennt berichtet werden.

Drei getrennte Prüfungen:

1. **Neue Evidenz unter unverändertem h:** Eine falsche/unsichere Hypothese wird
   durch korrekte neue Beispiele revidiert. Keine alten Labels widersprechen h.
2. **Belegberichtigung:** Ein absichtlich falsches Label wird mit `replace` repariert.
   Zwischenzeitlich darf die rauschfreie Hypothesenmenge leer sein; Oraclefehler
   sichtbar melden. Nach Reparatur muss Neuberechnung mit der bereinigten Referenz
   übereinstimmen. Modellqualität unter Korruption ist keine zugesicherte Fähigkeit.
3. **Persistenz:** Z, gültige Belege, Modellhash und Revision speichern; nach anderen
   Episoden wieder laden. Dieselben Abfragen reproduzieren dieselben Antworten
   innerhalb deklarierter numerischer Toleranz. Eine vorgegebene concept_id routet
   den Abruf; autonome Zuordnung/Abruf werden hier noch nicht geprüft.

Für einen kontrollierten informativen Beleg kann der Evaluator aus einem festen,
unbeschrifteten Kandidatenpool x* wählen, der das Disagreement konsistenter Regeln
maximiert, und y*=h(x*) offenbaren. Bei deterministischen Regeln entspricht das
maximaler binärer Entropie von p_O. Dies ist oraclegewählte Lehre, keine autonome
Informationssuche des Modells. Ein zweiter Updatearm erhält gleich viele zufällige
neue Belege aus demselben Pool. Gemeinsame Ausgangsgrößen `N0∈{4,8,16,32}` für jede Familie.
Vorgeschlagener Poolumfang 512, jeweils 1, 2 und
4 sukzessive Ergänzungen; Pools bleiben von Support und Q disjunkt. Alle Arme
erhalten dieselben modellunabhängig ausgewählten Ergänzungen. Ist der Pool für
eine Auswahl leer, Auswahlumfang und Zensierung melden; keine Episode verwerfen. Das Modell bekommt keinen Zugriff auf den Pool.

\[
 Gain_{corr}=Score(Q_{fixed},Z_{after})-Score(Q_{fixed},Z_{before}).
\]

`Q_fixed` bleibt von beiden Updatepools disjunkt. Eine zusätzliche modellabhängige
Fehlerauswahl ist explorativ: Ein informativer Beleg ändert den Oracleposterior;
ein Erinnerungsbeleg wird vom Oracle bereits eindeutig beantwortet, aber vom Modell
falsch. Gewinne getrennt ausweisen. Unterschiedliche, durch jeweilige Modellfehler
gewählte Belege erlauben keinen direkten kausalen Vergleich zwischen Armen. Neues Einlernen eines getrennten
Eintrags `c'` darf den gespeicherten Zustand und die Antworten von c nicht ändern.
Das ist hier eine Eigentums-/Routingkontrolle; durch getrennte Zustände wird noch
keine gelernte selektive Korrektur innerhalb eines einzigen gemischten Zustands gezeigt.

## 8. Kontrollen, Messwerte und Entscheidung

Kontrollen bei identischen Queryeingaben:

- `N=0` bzw. Support maskiert, inklusive Nullbeleg; Größenmetadaten dokumentieren.
- Code aus einer anderen Episode einsetzen; Regeln unterscheiden sich, und
  Auswertung zusätzlich auf Queries, bei denen die beiden Regeln widersprechen.
- Supportlabels in einer Episode permutieren; Labelzahl und Eingaben bleiben gleich.
  Eventuell weiterhin konsistente oder zufällig unveränderte Fälle separat zählen.
- Zustand s direkt kopieren sowie Mehrheitsantwort des deklarierten Priors.
- Bei kategorischen Regeln irrelevante Rollen/Zustandsbits verändern;
  bei Relationen Rollen tauschen und Antwort mit der wahren Regel erneut bestimmen.

Natürliche Query-BCE und Accuracy werden berichtet. Primärscore Kategorien/Relationen:
Balanced Accuracy über y=0 und y=1. Für Übergänge ist das interessante Ereignis
`Δ=y XOR s`, nicht bloß der nächste Zustand:

\[
 p_\Delta=\begin{cases}p& s=0\\1-p&s=1\end{cases},\quad
 \widehat\Delta=\mathbf1[p_\Delta\ge0.5],\quad
 BA_\Delta=\tfrac12\big(P(\widehat\Delta=1\mid\Delta=1)
                       +P(\widehat\Delta=0\mid\Delta=0)\big).
\]

Bei Gleichstand zunächst `y_hat = 1[logit >= 0]` setzen und dann
`delta_hat = y_hat XOR s` ableiten. Die Formel für `p_Δ` ist äquivalent außerhalb
exakter Gleichstände; bei Gleichständen gilt ausschließlich diese gemeinsame Regel. Scores zuerst pro
Episode/Regel berechnen, dann Regeln innerhalb der Familie gleich gewichten.
Leere Klassen als nicht auswertbar zählen; keine stillen Nullen/Ones einsetzen.

**Nachgewiesene Aufgabenfalle:** Bei uniformem x erreicht Kopieren von s bereits
87,5–93,75 % Accuracy. Daher reicht eine 90-%-Genauigkeitsschwelle für Übergänge
nicht. Kopieren erreicht im balancierten Änderungsmaß genau 50 %.

**Zweite, unabhängig bestätigte Aufgabenfalle:** Selbst nach der Änderungsbalancierung
kann ein Prädiktor, der nur den Operator und `(s,u)` kennt, gute Werte erreichen.
Er öffnet/schließt/toggelt bei jedem Eingriff und ignoriert das Matching vollständig:

\[
 F_{cat}=F_{rel}=\tfrac12,\quad
 F_{open}=F_{close}=\tfrac9{10},\quad F_{toggle}=\tfrac{11}{14},\quad
 F_{tr}=\tfrac{181}{210}.
\]

Hier sind F feste Referenzscores dieser expliziten Kurzschlussstrategien, keine
allgemeinen Informationsobergrenzen. Definiere normierte Kompetenz:

\[
 \nu_g=\frac{BA_g-F_g}{1-F_g},\qquad
 D_{g,r}=\nu^{A}_{g,r}-\nu^{C\_full}_{g,r}.
\]

Ein negativer ν-Wert wird nicht abgeschnitten. Die vorgeschlagenen Hauptgates bei
`N*=128` lauten für Kategorie, Relation und **jeden** der drei Übergangsoperatoren:

\[
 LCB_{.95}(\nu^A_g)\ge0.8,\qquad
 LCB_{.95}(D_g)\ge-0.03,\qquad
 LCB_{.95}(\nu^A_g-\nu^{A,empty}_g)>0.
\]

Die gleiche absolute Kompetenz muss C_full erreichen, damit ein schwacher Vergleich
keinen Erfolg vortäuscht. Die 0.8-Grenze entspricht BA `0.90` für Kategorie/Relation,
`0.98` für open/close und ungefähr `0.95714` für toggle. Zusätzlich wird die
Übergangsfamilie mit Floor `181/210` zusammengefasst (0.8 entspricht BA≈0.97238),
aber ihr Mittelwert darf keinen Operator verdecken. Die Nichtunterlegenheitsmarge
0.03 gilt auf der ν-Skala; sie ist keine pauschale Drei-Prozentpunkte-Marge auf BA.
Schwellen sind konkrete Vorschlagswerte, vor Trainingsbeginn im Manifest einzufrieren.
Ändern nach Testergebnissen erzeugt einen neuen Versuch. Unpräzise oder nicht
kompetente Vergleiche bleiben unentschieden, keine Widerlegung gemeinsamer Formate.

**Informationskalibrierung vor Modelltraining:** Mit Seed 9023 wurden pro Regel
20 uniforme Supportziehungen geprüft (5600 Episoden, vollständige 280-Regel-
Versionsmenge). Die erste geprüfte Supportgröße mit mindestens 90 % eindeutigen
Regeln betrug 16 für Kategorien, 16 für Relationen und 64 für Übergänge. Bei 32
Belegen waren 50,21 % der Übergangsregeln eindeutig, bei 64 waren es 93,39 %,
bei 128 waren es 99,974 %. Das ist eine endliche Generatorprüfung, kein gelerntes
Modellergebnis und keine Garantie für andere Daten. Der vorab festgelegte
Kalibrierungsentscheid `N*=2 max_f n90_f` ergibt **N*=128**. [Manifest und Rohwerte](../runs/reviews/shared_abstraction_formulas_20260923/oracle-calibration.json).

Vorschlagskonfiguration nach dieser Informationsprüfung:
Training `N=0` mit Wahrscheinlichkeit 0.05, sonst uniform aus `{4,8,16,32,64,128}`,
identisch für alle Familien. Evaluation bei `{0,4,8,16,32,64,128,256}` mit 20
Supportziehungen pro Testregel und `M=512` uniformen Queries, Hauptgate nur bei 128.
256 prüft Extrapolation. Jede Familie wird bei jeder Supportgröße geprüft;
unterschiedliche Größenverteilungen dürfen keine Familien-ID verraten.
Alle Testepisoden sind zwischen Armen und Trainingsseeds gepaart. Kein heimliches
Entfernen unbestimmter Supportmengen oder leerer Klassen. Eine zusätzlich simulierte
volle Grammatik unterscheidet Informationsgrenzen vom eng definierten Testsplit.

**Sekundäre Diagnose nach Claude:** Nur für die Auswertung definiere die
regelabhängige Teilmenge `R_h`: alle Eingaben für Kategorien/Relationen,
`{s=0,u=1}` für open, `{s=1,u=1}` für close und `{u=1}` für toggle.
Dort wird zusätzlich nach dem Matchingbit `m` balancierte Genauigkeit berichtet,
bei Kategorien nach y. Ein weiterer Score beschränkt sich auf durch `O_full`
eindeutig beantwortbare Queries. Seine Abdeckung je Regel und Klasse immer
mitberichten. Diese vom Evaluator gebildeten Teilmengen verändern weder die
Modellinputs noch den Hauptscore. Hohe bedingte Genauigkeit bei geringer Abdeckung
kann den vollständigen Erwerbs-/Nutzungsnachweis nicht ersetzen.

Die Hauptaussage und ihr Intervall gelten bedingt auf den vorab gespeicherten
Testregeln **und Testepisoden**. Queryzeilen sind keine Trainingsreplikate. Für fünf
gepaarte Trainingsseeds zunächst die vollständigen Scores pro Seed bilden, dann:

\[
 LCB_{.95}(D)=\bar D-t_{.95,R-1}\frac{s_D}{\sqrt R},\qquad
 s_D^2=\frac1{R-1}\sum_{r=1}^R(D_r-\bar D)^2,\quad R=5.
\]

Entsprechend für ν und Evidenzgewinn. Dies ist ein einseitiges gepaartes t-Intervall
mit angenommener näherungsweiser Normalität der unabhängigen Seedmittel; fünf Seeds
geben keine verteilungsfreie Absicherung. Alle Seedwerte veröffentlichen. Ein
zusätzliches Bootstrap über ganze Support-/Queryepisoden je fester Regel (gepaart
über Arme und Seeds) beschreibt Teststichprobenunsicherheit separat; das Seedintervall
darf nicht als diese zusätzliche Unsicherheit einschließend bezeichnet werden.
Ein einfacher Fünf-Seed-Perzentilbootstrap wird nicht als 95-%-Garantie ausgegeben.
Bei Intervallen, die die Marge nicht entscheiden, Ergebnis als unentschieden melden.

Eine Aussage über eine neue Regelpopulation benötigt mehr unabhängige Regelgruppen;
der Test enthält nur zwei zurückgehaltene Kategorie-Komplementgruppen. Gemeinsames
Bestehen aller vorher festgelegten Gruppen ist eine Konjunktionsentscheidung.
Nachträglich ausgesuchte Budgets, Teilgruppen oder neue Einzelaussagen sind nicht
von dieser Entscheidung abgedeckt. Vier Latin-Square-Folds verändern auch die
Trainingsdichte und wären ein neuer Versuchsplan, keine identische Replikation.

### Zweite Nutzung desselben Zustands

Ein positives Ergebnis des binären Ersttests reicht für die breitere Fragestellung
noch nicht. Folgende zusätzliche Anwendungen benötigen keine neue Repräsentation
oder Zielnetzwerke und sind unmittelbar mit dem definierten Leser ausführbar:

\[
 a^*=\arg\max_{a\in\{0,1,2,3\}^4}F_\theta(x(a,b,s,u),Z_{cat}),\qquad
 b^*=\arg\max_{b\in\{0,1,2,3\}^4}F_\theta(x(a,b,s,u),Z_{rel}).
\]

Bei Kategorien wird ein passendes Objekt konstruiert, bei Relationen zu einem
vorgegebenen a ein passendes b gesucht. Alle 256 Kandidaten werden geprüft;
G wird einmal pro Support aufgerufen, sein Z bleibt unverändert. Lexikografische
Tie-Breaks sind fest. Erfolg wird durch `h(a*)=1` bzw. `m(a,b*)=1` unabhängig
verifiziert, nicht durch Übereinstimmung mit nur einer willkürlichen Musterlösung.
Zufällige Auswahl und konstante Scores sind eigene Baselines. Suchkosten zählen
vollständig. Der monotone Sigmoidscore hat dieselbe Rangfolge wie der Logit.

Für Übergänge wird derselbe Zustand wiederholt angewendet:

\[
 \widehat s_{t+1}=\mathbf1[F_\theta(x(a,b,\widehat s_t,u_t),Z_{tr})\ge0],
 \qquad s_{t+1}=h(x(a,b,s_t,u_t)).
\]

Startzustand und Aktionsfolge werden vorab gezogen, `u_t` ist ein gesetzter Eingriff.
Prüfe Längen 1, 2 und 3, alle Zwischenzustände, freie Rollouts und zusätzlich
Einzelschritte mit wahrem s_t. Nur den Endzustand zu bewerten wäre falsch: zweimaliges
Toggle mit gleichem Matching kehrt immer zum Anfang zurück. Kopieren und andere
Kurzschlussstrategien können dadurch gute Endpunktwerte erhalten.

Diese Anwendungen prüfen die Nutzbarkeit desselben Z für Auswahl und wiederholte
Anwendung. Enumeration und Schleife sind programmiert; sie beweisen keine gelernte
Suchstrategie, freie Generierung oder neue Kompositionsoperation. Eine stärkere
Fortsetzung trainiert einen gemeinsamen Kern für mehrere Abfrageoperationen und
hält Kombinationen aus Familie und Operation zurück. Sie wäre ein neues Experiment.

## 9. Umsetzung als kleine Recipe

Keine neue Trainer-/Registry-/CLI-Hierarchie. Eine normale Experimentrecipe baut
Datengenerator, bestehende einfache PyTorch-Bausteine, Loss und Vergleichsarme.
Tensorverträge oben sind die Schnittstelle; Code darf keine Modellinputs aus
Evaluatorwissen konstruieren.

```python
for batch in train_episodes(manifest):
    # y_query does not enter infer() or predict().
    z = model.infer(batch.support_x, batch.support_y, batch.support_valid)
    logits = model.predict(batch.query_x, z)
    loss = family_mean_weighted_bce(
        logits, batch.query_y, batch.weights_for_loss_only, batch.family_for_loss_only
    )
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

freeze_parameters_and_buffers(model)
for episode in test_episodes(manifest):
    z = model.infer(episode.support_x, episode.support_y, episode.support_valid)
    before = model.predict(episode.fixed_queries, z)
    store.apply(episode.explicit_update)
    support = store.active_evidence(episode.concept_id)
    z_new = model.infer(*support.as_tensors())
    after = model.predict(episode.fixed_queries, z_new)
    evaluator.record(before, after, hidden_truth=episode.targets)
```

Wesentliche Implementierungschecks: boolesche Wahrheitstabelle unabhängig prüfen;
Splitgruppen und Support/Query-Disjunktheit; endliche Outputs bei leerem Support;
Permutation/Padding ändern Vorhersagen nur innerhalb Toleranz; Querybatching ändert
keine Einzelantwort; Querylabels erreichen keinen Inferenzpfad; Gradienten erreichen
G im Training; Parameter-/Bufferhash bleibt in Evaluation gleich; Replacement und
Neuberechnung stimmen überein; Save/Load reproduziert Antworten. Das sind später
Implementierungschecks, kein Ersatz für einen erfolgreichen Lernversuch.

Stehende Prinzipien: Belege einmal vorbereiten, Zustand mehrfach lesen; Evidenz
und daraus abgeleitete Zustände getrennt besitzen/versionieren; Informationsverlust
und Speicher-/Zugriffskosten offen messen; identische geteilte Tiefe im Training
und Test; überprüfbare Ausgaben statt bloßer Latentähnlichkeit. Pixel, multimodale
Adapter, gelernte Auswahl von Belegen und adaptive Tiefe folgen erst als eigene
Faktoren. Das hält den ersten Versuch diagnostizierbar und die Bibliothek klein.

## 10. Review, Belege und offene Grenzen

Zwei tatsächliche Claude-Aufrufe sind abgeschlossen. Beide Ausführungsbelege
bestätigen `claude-opus-5-5` und explizites `--effort max`; keine Tools oder
zusätzlichen Modelle. Die Anfrage enthielt ausschließlich eigens formulierte,
öffentlich teilbare methodische Fragen und die generische Grammatik, keinen
privaten Quellcode oder Projektmesswerte. Claude kann seinen Denkaufwand nicht
selbst verifizieren; maßgeblich sind Aufrufparameter und Modellbelege.

**Gemeinsam geklärt und umgesetzt:**

- Architekturteilung, kompatible Codes und gemeinsame innere Semantik unterscheiden.
- Familieninformation mit A+ID kontrollieren; Leser- und Encoderspezialisierung
  durch B und C_full getrennt untersuchen und Parameterunterschiede offenlegen.
- Alle Queries als Hauptscore beibehalten, Identifizierbarkeit und regelabhängige
  Teilmengen als zusätzliche Diagnosen mit Abdeckungswerten ausweisen.
- Gewichtete BCE und U-Entscheidungsregel sind algebraisch konsistent. `p` ist ein
  Entscheidungsscore; gewichtetes CE-Regret gegen die jeweilige q_O-Referenz ist
  eine mögliche zusätzliche Diagnose, keine natürliche Wahrscheinlichkeitskalibrierung.
- Die zweite Kurzschlussstrategie verlangt normierte Kompetenzgates; Floors und
  resultierende Schwellen wurden unabhängig über die komplette Eingabedomäne bestätigt.
- Feste Speicherkapazität nicht mit Informationskompression gleichsetzen;
  Persistenz/Recompute nicht als gelerntes Gedächtnisupdate ausgeben.

**Von Claude zurückgenommen/eingegrenzt:** Ein gelerntes natives Operatorformat
ist kein exaktes Inferenzoracle; unabhängig trainierte Codes müssen nicht prinzipiell
inkompatibel sein; einzelne Duplikate verändern relative Attentiongewichte; O_full
ist bei abweichendem Prior keine allgemeine Leistungsobergrenze. Der zunächst
vorgeschlagene nur auf identifizierte Queries beschränkte Hauptscore wurde im
Rückabgleich zugunsten des vollständigen Hauptscores aufgegeben. Die frühere
Priorunabhängigkeit gilt nicht für den neuen gewichteten Hauptscore.

**Eigene Prüfungen:** 280 unterschiedliche Funktionen über 262144 Eingaben;
143360 skalare Gegenprüfungen und ebenso viele native Darstellbarkeitsprüfungen;
endliche Nullsupportwerte, Gradientenpfade, Permutations-/Paddingkontrollen und
Querybatching mit zufälligen Gewichten. Informationskalibrierung mit 5600 Episoden.
Die zusätzliche Auswertung der gewichteten Referenzentscheidung bei N=128 erreichte
in den geprüften je 512 Queries pro Episode ν=1 in allen fünf Gruppen; damit sind
die vorgeschlagenen 0.8-Gates in dieser Stichprobe informationell erreichbar.
Dies ist kein trainiertes Modell und keine Garantie außerhalb dieser Grammatik.

Noch offen sind tatsächliche Lernbarkeit, Konvergenz bei 5000 Updates, gemessene
GPU-/Laufzeitkosten, Empfindlichkeit gegen Hyperparameter und weiter entfernte
Abstraktionsarten. Weitere Folds, Quantisierung, eigenständige Rollenbindung,
Multimodalität und gelernte Komposition bleiben neue Versuche. Die Revieweinigkeit
wählt keine Produktionsarchitektur und ersetzt keinen Lernnachweis.

[Dauerhafter Prüfbeleg](../ara/evidence/tables/shared_abstraction_spec_review_2026-09-23.json).
[Erster Review](../runs/reviews/shared_abstraction_formulas_20260923/review.md),
[Rückabgleich](../runs/reviews/shared_abstraction_formulas_20260923/reconciliation.md),
[rohe Generator-/Formelprüfungen und Bericht](../runs/reviews/shared_abstraction_formulas_20260923/report.html).
Der vorhandene Reportrenderer wurde verwendet; der Bericht ist strukturell als
standalone geprüft, ohne Browserprüfung. Resultat- und Berichtsstatus sind getrennt.
