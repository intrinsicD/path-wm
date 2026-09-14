/** Render the hand-audited architecture map; no model execution or network use.
 * Usage: node docs/diagrams/render_architecture_atlas.mjs [path/to/@viz-js/viz/dist/viz.js]
 * Requires @viz-js/viz (rendered with 3.25.0). This is documentation glue, not a model tracer.
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve, relative } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const directory = dirname(fileURLToPath(import.meta.url));
const root = resolve(directory, '../..');
const spec = JSON.parse(await readFile(resolve(directory, 'architecture-atlas.json'), 'utf8'));
const { instance } = await import(process.argv[2]
  ? pathToFileURL(resolve(process.argv[2])).href : '@viz-js/viz');
const viz = await instance();
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const quote = JSON.stringify;
// The bundled Graphviz font-metrics table covers ASCII. Keep rendered layout
// deterministic; Unicode remains in the editable Mermaid and surrounding text.
const graphText = value => value.replace(/[·→×Δτ√ᵀ²≤§–—←]/g, c => ({
  '·': '/', '→': '->', '×': 'x', 'Δ': 'delta ', 'τ': 'tau',
  '√': 'sqrt ', 'ᵀ': '^T', '²': '^2', '≤': '<=', '§': 'section ', '–': '-', '—': '-', '←': '<-',
}[c]));
const wrap = (value, limit) => value.split('\n').map(line => {
  const lines = [''];
  for (const word of line.split(' ')) {
    const last = lines.length - 1;
    if (lines[last] && lines[last].length + word.length + 1 > limit) lines.push(word);
    else lines[last] += (lines[last] ? ' ' : '') + word;
  }
  return lines.join('\n');
}).join('\n');
const styles = {
  learned: { fill: '#e6eef8', color: '#7696bc', label: 'Learned module' },
  store: { fill: '#f3f4f6', color: '#9098a4', label: 'Explicit state / mechanics' },
  external: { fill: '#e7f1eb', color: '#789887', label: 'External input / output' },
  optional: { fill: '#efeafa', color: '#9c87b5', label: 'Optional experiment / path' },
  training: { fill: '#fff0db', color: '#bd934d', label: 'Training / diagnostic only' },
  proposal: { fill: '#fafafa', color: '#9b9b9b', label: 'Proposed extension (dashed)' },
};
const sourceCache = new Map();
async function source(reference) {
  const [path, symbol] = reference.split('#');
  if (!sourceCache.has(path)) sourceCache.set(path, await readFile(resolve(root, path), 'utf8'));
  const text = sourceCache.get(path);
  const index = symbol ? text.split('\n').findIndex(line =>
    new RegExp(`^\\s*(?:class|def|async def) ${symbol}\\b`).test(line)) : -1;
  if (symbol && index < 0) throw new Error(`Missing source symbol: ${reference}`);
  return { path, symbol, line: index + 1, label: symbol ? `${path} · ${symbol}:${index + 1}` : path };
}

const outputDirectory = resolve(directory, 'atlas');
await mkdir(outputDirectory, { recursive: true });
const rendered = [];
for (const graph of spec.graphs) {
  if (!/^\d{2}-[a-z-]+$/.test(graph.id)) throw new Error(`Invalid graph ID: ${graph.id}`);
  const ids = new Set(graph.nodes.map(n => n[0]));
  if (ids.size !== graph.nodes.length) throw new Error(`Duplicate node: ${graph.id}`);
  const dot = [
    'digraph G {',
    `graph [rankdir=${graph.direction}, bgcolor="white", pad=0.2, nodesep=0.28, ranksep=0.55, splines=polyline, outputorder=edgesfirst];`,
    'node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=14, margin="0.18,0.13", penwidth=1.1];',
    'edge [fontname="DejaVu Sans", fontsize=11, color="#6b7280", fontcolor="#374151", arrowsize=0.7, penwidth=1.1];',
  ];
  const mermaid = [`flowchart ${graph.direction}`];
  for (const [id, label, kind] of graph.nodes) {
    if (!/^[a-z][a-z0-9_]*$/.test(id) || !styles[kind]) throw new Error(`Invalid node ${id}`);
    const style = styles[kind];
    dot.push(`${quote(id)} [label=${quote(wrap(graphText(label), 46))}, fillcolor=${quote(style.fill)}, color=${quote(style.color)}${kind === 'proposal' ? ', style="rounded,filled,dashed"' : ''}];`);
    mermaid.push(`    ${id}["${escape(label).replaceAll('\n', '<br/>')}"]`);
    mermaid.push(`    class ${id} ${kind};`);
  }
  for (const [from, to, label, kind] of graph.edges) {
    if (!ids.has(from) || !ids.has(to)) throw new Error(`Dangling edge ${graph.id}: ${from} -> ${to}`);
    const dashed = kind === 'proposal' || kind === 'training';
    const feedback = graph.feedback_edges?.some(([a, b]) => a === from && b === to);
    dot.push(`${quote(from)} -> ${quote(to)} [label=${quote(wrap(graphText(label), 32))}${dashed ? ', style=dashed' : ''}${kind === 'training' ? ', color="#b88a3e"' : ''}${feedback ? ', constraint=false' : ''}];`);
    mermaid.push(`    ${from} ${dashed ? '-.->' : '-->'}|"${escape(label)}"| ${to}`);
  }
  dot.push('}');
  for (const [kind, style] of Object.entries(styles)) {
    mermaid.push(`    classDef ${kind} fill:${style.fill},stroke:${style.color},color:#202a36${kind === 'proposal' ? ',stroke-dasharray:5 4' : ''};`);
  }
  const result = viz.render(dot.join('\n'), { format: 'svg', engine: 'dot' });
  if (result.status !== 'success' || result.errors.length) {
    throw new Error(`Graphviz ${graph.id}: ${JSON.stringify(result.errors)}`);
  }
  const svg = result.output.slice(result.output.indexOf('<svg'))
    .replace(/id="([^"]+)"/g, (_, id) => `id="${graph.id}-${id}"`);
  const sources = await Promise.all(graph.sources.map(source));
  const svgPath = resolve(outputDirectory, `${graph.id}.svg`);
  await writeFile(svgPath, svg);
  const viewBox = svg.match(/viewBox="([^"]+)"/)[1].split(' ').map(Number);
  rendered.push({ ...graph, sources, svg, mermaid: mermaid.join('\n'), width: viewBox[2], height: viewBox[3] });
}

const markdown = [
  '# ' + spec.title, '', spec.intro, '',
  `Source review: ${spec.date}, repository snapshot \`${spec.source_commit}\`. [Open the rendered atlas](architecture-atlas.html).`, '',
  'Blue: learned modules. Gray: explicit state/mechanics. Green: external I/O. Purple: optional path. Amber: training/control only. Dashed: proposed extension or a labelled training-only connection. Colors describe roles, not measured competence.', '',
  ...rendered.map(g => `- [${g.title}](#${g.id})`), '',
];
for (const g of rendered) markdown.push(
  `<a id="${g.id}"></a>`, '', '## ' + g.title, '', g.subtitle, '',
  '```mermaid', g.mermaid, '```', '',
  `[Full-size SVG](diagrams/atlas/${g.id}.svg)`, '',
  ...g.notes.flatMap(note => [note, '']),
  'Source: ' + g.sources.map(s => `[${s.label}](../${s.path})`).join(', ') + '.', '',
);
await writeFile(resolve(root, 'docs/architecture-atlas.md'), markdown.join('\n'));

const legend = Object.values(styles).map(s => `<span><i style="background:${s.fill};border-color:${s.color}"></i>${escape(s.label)}</span>`).join('');
const sections = rendered.map((g, index) => `<details id="${g.id}"${index === 0 ? ' open' : ''}>
<summary><span>${escape(g.title)}</span><small>${escape(g.subtitle)}</small></summary>
<div class="section-body"><div class="diagram" role="img" aria-label="${escape(g.title + '. ' + g.subtitle)}">${g.svg.replace('<svg ', `<svg style="min-width:${Math.ceil(g.width * 0.8)}px" `)}</div>
<p class="links"><a href="diagrams/atlas/${g.id}.svg">Open full-size drawing</a> · <a href="#top">Back to map</a></p>
${g.notes.map(note => `<p>${escape(note)}</p>`).join('\n')}
<details class="sources"><summary>Implementation sources</summary><ul>${g.sources.map(s => `<li><a href="../${escape(s.path)}">${escape(s.label)}</a></li>`).join('')}</ul></details>
</div></details>`).join('\n');
const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escape(spec.title)}</title>
<style>
:root { color-scheme:light; font:16px/1.55 system-ui,sans-serif; color:#202a36; background:#f8f9fb; }
* { box-sizing:border-box; } body { margin:0; } main { max-width:1320px; margin:auto; padding:38px 32px 70px; }
h1 { font-size:32px; line-height:1.2; margin:8px 0 18px; } .eyebrow { font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:#536478; }
header>p { max-width:960px; } a { color:#245c98; text-underline-offset:3px; } a:focus-visible,button:focus-visible,summary:focus-visible { outline:3px solid #286cb0; outline-offset:3px; }
.legend { display:flex; flex-wrap:wrap; gap:10px 20px; font-size:13px; margin:22px 0; } .legend span { display:flex; align-items:center; gap:7px; }
.legend i { width:17px; height:12px; border:1px solid; border-radius:3px; }
nav { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px 22px; margin:22px 0 24px; } nav a { font-size:14px; }
.controls { display:flex; gap:10px; margin:22px 0; } button { font:inherit; background:#fff; color:#27435d; border:1px solid #bcc9d5; padding:7px 12px; border-radius:5px; cursor:pointer; }
main>details { background:#fff; border:1px solid #d7dfe7; border-radius:8px; margin:14px 0; scroll-margin-top:16px; }
summary { cursor:pointer; padding:18px 22px; } summary>span { font-size:20px; font-weight:600; } summary small { display:block; color:#596979; padding-left:18px; font-size:13px; }
.section-body { padding:0 22px 22px; } .diagram { overflow:auto; padding:14px 0 6px; } .diagram svg { display:block; width:100%; min-width:660px; height:auto; }
.section-body>p { max-width:1080px; } .links { font-size:13px; } .sources { border-top:1px solid #e4e8ee; margin-top:18px; font-size:13px; } .sources summary { padding:12px 0 0; } .sources li { overflow-wrap:anywhere; }
footer { color:#596979; font-size:13px; margin-top:28px; }
@media(max-width:760px) { main { padding:22px 12px 40px; } h1 { font-size:27px; } nav { grid-template-columns:1fr; } summary { padding:14px; } summary>span { font-size:18px; } .section-body { padding:0 12px 16px; } }
@media print { :root { background:white; } main { max-width:none; padding:0; } .controls,nav,.links { display:none; } main>details { break-before:page; border:0; } .diagram svg { min-width:0; } summary { list-style:none; } }
</style></head><body><main id="top">
<header><div class="eyebrow">Architecture review · ${escape(spec.date)}</div><h1>${escape(spec.title)}</h1>
<p>${escape(spec.intro)}</p><p>Open a level below to inspect its inputs, outputs and inner workings. Sources and limitations accompany every drawing.</p></header>
<div class="legend" aria-label="Diagram legend">${legend}</div>
<nav aria-label="Architecture levels">${rendered.map(g => `<a href="#${g.id}">${escape(g.title)}</a>`).join('')}</nav>
<div class="controls"><button type="button" id="expand">Expand all diagrams</button><button type="button" id="collapse">Collapse details</button></div>
${sections}
<footer>Grounded in repository snapshot ${escape(spec.source_commit)}. Static diagrams are manually audited descriptions, not execution traces. Model code and weights are unchanged. <a href="architecture-atlas.md">Mermaid source and notes</a>.</footer>
</main><script>
const panels = [...document.querySelectorAll('main > details')];
document.getElementById('expand').addEventListener('click', () => panels.forEach(p => p.open = true));
document.getElementById('collapse').addEventListener('click', () => panels.forEach((p,i) => p.open = i === 0));
function reveal() { const panel = document.getElementById(location.hash.slice(1)); if (panel?.matches('main > details')) { panel.open = true; panel.scrollIntoView(); } }
document.querySelectorAll('nav a').forEach(a => a.addEventListener('click', () => { const panel = document.getElementById(a.hash.slice(1)); panel.open = true; }));
window.addEventListener('hashchange', reveal); reveal();
window.addEventListener('beforeprint', () => panels.forEach(p => p.open = true));
</script></body></html>`;
await writeFile(resolve(root, 'docs/architecture-atlas.html'), html);
console.log(JSON.stringify({ graphs: rendered.map(g => ({ id:g.id, nodes:g.nodes.length, edges:g.edges.length, width:g.width, height:g.height })), sources:sourceCache.size, output:relative(root, resolve(root, 'docs/architecture-atlas.html')) }, null, 2));
