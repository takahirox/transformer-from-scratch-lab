import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {copyFile, lstat, mkdir, readFile, rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export const MAX_SITE_BYTES = 50 * 1024 * 1024;
const staticFiles = ['index.html', 'style.css', 'app.js', 'core.js', 'runtime.js',
  'THIRD_PARTY_NOTICES.txt'];
const runtimeFiles = ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm', 'ThirdPartyNotices.txt'];

function modelFile(name) {
  assert.equal(typeof name, 'string', 'Missing model filename');
  assert.match(name, /^[\w.-]+\.(onnx|json)$/, 'Model paths must be local filenames');
  return `model/${name}`;
}

export async function packagePages(source, target, maxBytes = MAX_SITE_BYTES) {
  // Never leave a previous successful artifact after a failed rebuild.
  await rm(target, {recursive: true, force: true});
  const manifest = JSON.parse(await readFile(path.join(source, 'model/manifest.json'), 'utf8'));
  assert.equal(manifest.format_version, 1);
  assert.deepEqual(manifest.models.map(entry => entry.length), [2, 16, 32, 64],
    'Pages requires all four exported lengths');
  assert.ok(manifest.training_config.steps > 0, 'A trained checkpoint is required');
  const referenceFile = modelFile(manifest.reference);
  const reference = JSON.parse(await readFile(path.join(source, referenceFile), 'utf8'));
  assert.equal(reference.cases.length, manifest.models.length * 3);
  const {classes, heads} = manifest.model_settings;
  for (const entry of manifest.models) {
    const data = await readFile(path.join(source, modelFile(entry.file)));
    assert.equal(data.length, entry.bytes, `Model size mismatch: ${entry.file}`);
    assert.equal(createHash('sha256').update(data).digest('hex'), entry.sha256,
      `Model hash mismatch: ${entry.file}`);
    const cases = reference.cases.filter(test => test.length === entry.length);
    assert.equal(cases.length, 3, `Missing references for length ${entry.length}`);
    for (const test of cases) {
      assert.equal(test.tokens.length, entry.length);
      assert.equal(test.logits.length, classes);
      assert.equal(test.attention.length, heads * entry.length ** 2);
    }
  }
  const files = [...staticFiles, ...runtimeFiles.map(name => `vendor/${name}`),
    'model/manifest.json', referenceFile, ...manifest.models.map(entry => modelFile(entry.file))];
  assert.equal(new Set(files).size, files.length, 'Duplicate artifact paths');
  let bytes = 0;
  for (const file of files) {
    const stat = await lstat(path.join(source, file));
    assert.ok(stat.isFile() && stat.size > 0, `Missing/empty regular file: ${file}`);
    bytes += stat.size;
  }
  assert.ok(bytes <= maxBytes, `Pages site too large: ${bytes} bytes > ${maxBytes} bytes`);
  // Check literal imports and HTML/CSS asset references in our own source.
  // Runtime-internal JS/WASM dependencies are explicitly listed above and
  // fetched in the subpath browser smoke test.
  for (const file of staticFiles.filter(name => /\.(html|js|css)$/.test(name))) {
    const content = await readFile(path.join(source, file), 'utf8');
    const patterns = [/\b(?:src|href)=["']([^"']+)["']/g,
      /(?:\bfrom\s*|\bimport\s*\(\s*)["']([^"']+)["']/g,
      /url\(["']?([^\s)'";]+)["']?\)/g];
    for (const pattern of patterns) {
      for (const [, url] of content.matchAll(pattern)) {
        if (url.startsWith('#')) continue;
        assert.ok(url.startsWith('./'), `Expected a relative asset URL in ${file}: ${url}`);
        assert.ok(files.includes(path.posix.normalize(url)), `Unpackaged asset in ${file}: ${url}`);
      }
    }
  }
  // Allowlist copies only: no checkpoints, training reports, node_modules or
  // arbitrary source files, even when they exist in the source directories.
  for (const file of files) {
    await mkdir(path.dirname(path.join(target, file)), {recursive: true});
    await copyFile(path.join(source, file), path.join(target, file));
  }
  return {files: files.length, bytes, maxBytes};
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const source = fileURLToPath(new URL('../', import.meta.url));
  const target = fileURLToPath(new URL('../../results/pages/transformer-from-scratch-lab/', import.meta.url));
  const result = await packagePages(source, target);
  console.log(`Pages site: ${result.files} files, ${result.bytes} bytes ` +
    `(${(result.bytes / 1024 ** 2).toFixed(2)} MiB); limit ${result.maxBytes} bytes.\n${target}`);
}
