#!/usr/bin/env node
/** Canonical report delivery with a narrowly scoped shared-reader sizing repair.
 * The bundled toolbar uses 100vw, which includes a classic scrollbar and causes
 * real horizontal overflow. Use the shell width plus its own gutters instead.
 * No generated HTML, payload, renderer logic or verifier assertion is rewritten.
 */
import { createHash } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [pluginRoot, inputPath, outputPath] = process.argv.slice(2);
if (!pluginRoot || !inputPath || !outputPath) {
  throw new Error('Usage: node scripts/deliver_perception_proposal.mjs <plugin-root> <artifact.json> <report.html>');
}
const scriptRoot = resolve(pluginRoot, 'skills/build-report/scripts');
const { buildPortableArtifact, readPackagedReaderRuntime } = await import(pathToFileURL(resolve(scriptRoot, 'build_portable_artifact.mjs')).href);
const { deliverPortableArtifact } = await import(pathToFileURL(resolve(scriptRoot, 'deliver_portable_artifact.mjs')).href);
const original = readPackagedReaderRuntime().html;
let matches = 0;
const repaired = original.replace(/\.analytics-top-bar\s*\{[^}]+\}/g, rule => {
  if (!/width:\s*100vw/.test(rule)) return rule;
  matches += 1;
  return rule.replace(/width:\s*100vw/, 'width:calc(100% + 2 * var(--ds-gutter))')
    .replace(/margin-right:\s*calc\(50%\s*-\s*50vw\)/, 'margin-right:calc(-1 * var(--ds-gutter))')
    .replace(/margin-left:\s*calc\(50%\s*-\s*50vw\)/, 'margin-left:calc(-1 * var(--ds-gutter))');
});
if (matches !== 1) throw new Error(`Expected one known toolbar rule; found ${matches}. Review the updated reader before packaging.`);
process.env.CHROMIUM_EXECUTABLE_PATH ??= resolve('viewer/chromium_transport.mjs');
const receipt = await deliverPortableArtifact({ inputPath, outputPath }, {
  build: (input, options = {}) => buildPortableArtifact(input, { ...options, runtimeHtml: repaired }),
});
receipt.runtimeSizingRepair = {
  originalSha256: createHash('sha256').update(original).digest('hex'),
  packagedSha256: createHash('sha256').update(repaired).digest('hex'),
  affectedRule: '.analytics-top-bar',
  reason: 'Container-relative toolbar sizing avoids including scrollbar width; canonical verification is unchanged.',
};
writeFileSync(outputPath.replace(/\.html$/, '.receipt.json'), JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt));
if (!receipt.ok || receipt.stages?.verification !== 'passed') process.exitCode = 1;
