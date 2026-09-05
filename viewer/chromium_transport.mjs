#!/usr/bin/env node
/** Real-time transport for canonical Data Analytics Chromium probes.
 * Chrome virtual-time dump-DOM can starve requestAnimationFrame startup.
 * Execute the unchanged probe through a private CDP pipe, returning actual DOM.
 * No artifact, renderer, assertion or probe result is modified here.
 * CDP: https://chromedevtools.github.io/devtools-protocol/
 */
import { spawn } from 'node:child_process';
import { existsSync, readdirSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

function browserPath() {
  if (process.env.PATH_WM_CHROMIUM) return process.env.PATH_WM_CHROMIUM;
  if (existsSync('/opt/google/chrome/chrome')) return '/opt/google/chrome/chrome';
  const root = join(homedir(), '.cache/puppeteer/chrome-headless-shell');
  if (existsSync(root)) {
    for (const version of readdirSync(root).sort().reverse()) {
      const candidate = join(root, version, 'chrome-headless-shell-linux64/chrome-headless-shell');
      if (existsSync(candidate)) return candidate;
    }
  }
  throw new Error('No installed Chromium found; set PATH_WM_CHROMIUM');
}

async function main() {
  const args = process.argv.slice(2);
  const option = name => args.find(a => a.startsWith(`--${name}=`))?.split('=').slice(1).join('=');
  const url = args.find(a => a.startsWith('file:'));
  if (!url) throw new Error('Browser probe requires a local file URL');
  const [width, height] = (option('window-size') ?? '1440,1000').split(',').map(Number);
  const budget = Number(option('virtual-time-budget') ?? 8000);
  if (![width, height, budget].every(n => Number.isFinite(n) && n > 0)) {
    throw new Error('Invalid browser probe viewport or deadline');
  }
  const screenshot = option('screenshot');
  const browserArgs = args.filter(a => a !== url && a !== '--dump-dom' &&
    !a.startsWith('--virtual-time-budget=') && !a.startsWith('--screenshot='));
  const child = spawn(browserPath(), [...browserArgs,
    '--headless', '--no-sandbox', '--disable-gpu', '--remote-debugging-pipe',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows', 'about:blank'],
    { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
  const pending = new Map();
  let counter = 0, buffer = '', diagnostics = '';
  const rejectAll = error => {
    for (const { reject } of pending.values()) reject(error);
    pending.clear();
  };
  child.stderr.on('data', chunk => { diagnostics = (diagnostics + chunk).slice(-4000); });
  child.on('error', rejectAll);
  child.on('exit', code => rejectAll(new Error(`Chromium exited (${code}): ${diagnostics}`)));
  child.stdio[4].setEncoding('utf8');
  child.stdio[4].on('data', chunk => {
    buffer += chunk;
    let end;
    while ((end = buffer.indexOf('\0')) >= 0) {
      const message = JSON.parse(buffer.slice(0, end));
      buffer = buffer.slice(end + 1);
      const request = pending.get(message.id);
      if (!request) continue;
      pending.delete(message.id);
      if (message.error) request.reject(new Error(JSON.stringify(message.error)));
      else request.resolve(message.result);
    }
  });
  const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++counter;
    pending.set(id, { resolve, reject });
    child.stdio[3].write(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }) + '\0');
  });
  const watchdog = setTimeout(() => {
    rejectAll(new Error('Browser probe transport deadline exceeded'));
    child.kill('SIGKILL');
  }, budget + 2000);
  try {
    const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
    const command = (method, params) => send(method, params, sessionId);
    await command('Page.enable');
    await command('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
    const dark = option('blink-settings')?.includes('preferredColorScheme=0');
    await command('Emulation.setEmulatedMedia', { features: [
      { name: 'prefers-color-scheme', value: dark ? 'dark' : 'light' },
      { name: 'prefers-reduced-motion', value: 'reduce' },
    ] });
    const navigation = await command('Page.navigate', { url });
    if (navigation.errorText) throw new Error(navigation.errorText);
    const deadline = Date.now() + budget;
    let complete = false;
    while (Date.now() < deadline) {
      const result = await command('Runtime.evaluate', {
        expression: screenshot
          ? `document.documentElement.dataset.dataAnalyticsPortableReader === 'ready'`
          : `Boolean(document.querySelector('meta#data-analytics-portable-verifier-result,meta[data-portable-chart-extraction]'))`,
        returnByValue: true,
      });
      if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
      if (result.result.value) { complete = true; break; }
      await delay(25);
    }
    if (screenshot) {
      const capture = await command('Page.captureScreenshot', { format: 'png' });
      writeFileSync(screenshot, Buffer.from(capture.data, 'base64'));
    }
    if (!complete) throw new Error(`Canonical browser probe did not return a result within ${budget}ms`);
    const dom = await command('Runtime.evaluate', {
      expression: 'document.documentElement.outerHTML', returnByValue: true,
    });
    if (dom.exceptionDetails || typeof dom.result.value !== 'string') throw new Error('Browser probe DOM unavailable');
    process.stdout.write(dom.result.value + '\n');
  } finally {
    clearTimeout(watchdog);
    child.kill('SIGKILL');
  }
}

main().catch(error => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
