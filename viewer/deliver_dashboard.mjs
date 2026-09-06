#!/usr/bin/env node
/** Keep canonical packaging/QA; correct its reader's scrollbar-width header.
 * The installed 0.2.10 reader uses 100vw, which includes desktop scrollbars.
 * Supply its own runtime with one layout rule through the public builder API.
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const args = process.argv.slice(2);
const option = name => args[args.indexOf(name) + 1];
try {
  const builder = option('--builder');
  const { deliverPortableArtifact } = await import(pathToFileURL(builder));
  const { buildPortableArtifact, readPackagedReaderRuntime } = await import(
    pathToFileURL(join(dirname(builder), 'build_portable_artifact.mjs')));
  const css = readFileSync(new URL('./portable_layout.css', import.meta.url), 'utf8');
  const runtime = readPackagedReaderRuntime().html;
  // Bundled script strings also contain </head>; target the document terminator.
  const headEnd = runtime.lastIndexOf('</head>');
  if (headEnd < 0) throw new Error('Canonical reader has no closing head');
  const runtimeHtml = runtime.slice(0, headEnd) +
    `<style data-path-wm-layout-repair>${css}</style>` + runtime.slice(headEnd);
  const receipt = await deliverPortableArtifact({
    inputPath: option('--input'), outputPath: option('--output'),
    // Public canonical options: allow bounded startup under shared CPU load.
    readyTimeoutMs: 10000, actionTimeoutMs: 5000, timeoutMs: 25000,
  }, {
    build: (input, options = {}) => buildPortableArtifact(input, { ...options, runtimeHtml }),
  });
  process.stdout.write(JSON.stringify(receipt) + '\n');
} catch (error) {
  process.stderr.write(JSON.stringify(error.deliveryResult ?? { ok: false, error: error.message }) + '\n');
  process.exitCode = 1;
}
