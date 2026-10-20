import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, mkdir, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {test} from 'node:test';
import {packagePages} from '../scripts/package-pages.mjs';

test('Pages packaging rejects missing, corrupt and oversized bundles and copies only required files', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'recall-pages-'));
  const source = path.join(root, 'source');
  const target = path.join(root, 'site');
  try {
    await mkdir(path.join(source, 'model'), {recursive: true});
    await mkdir(path.join(source, 'vendor'));
    const models = [2, 16, 32, 64].map(length => ({length, file: `recall-${length}.onnx`,
      bytes: 5, sha256: createHash('sha256').update('model').digest('hex')}));
    const manifest = {format_version: 1, models, reference: 'reference.json',
      training_config: {steps: 200}, model_settings: {classes: 8, heads: 4}};
    const cases = models.flatMap(({length}) => Array.from({length: 3}, () => ({length,
      tokens: Array(length).fill(0), logits: Array(8).fill(0), attention: Array(4 * length ** 2).fill(0)})));
    await writeFile(path.join(source, 'model/manifest.json'), JSON.stringify(manifest));
    await writeFile(path.join(source, 'model/reference.json'), JSON.stringify({cases}));
    for (const entry of models) await writeFile(path.join(source, `model/${entry.file}`), 'model');
    const files = ['index.html', 'style.css', 'app.js', 'core.js', 'runtime.js', 'THIRD_PARTY_NOTICES.txt',
      'vendor/ort.webgpu.min.mjs', 'vendor/ort-wasm-simd-threaded.jsep.mjs',
      'vendor/ort-wasm-simd-threaded.jsep.wasm', 'vendor/ThirdPartyNotices.txt',
      'model/private-checkpoint.pt', 'vendor/unneeded.wasm'];
    for (const file of files) await writeFile(path.join(source, file), 'fixture');
    const result = await packagePages(source, target);
    assert.equal(result.files, 16);
    assert.ok(result.bytes > 0);
    assert.equal((await readdir(path.join(target, 'vendor'))).length, 4);
    assert.equal((await readdir(path.join(target, 'model'))).length, 6);
    await assert.rejects(packagePages(source, target, result.bytes - 1), /too large/);
    await assert.rejects(readFile(path.join(target, 'index.html')), /ENOENT/);
    await writeFile(path.join(source, 'model/recall-2.onnx'), 'wrong');
    await assert.rejects(packagePages(source, target), /hash mismatch/);
    await writeFile(path.join(source, 'model/recall-2.onnx'), 'model');
    await rm(path.join(source, 'vendor/ort-wasm-simd-threaded.jsep.wasm'));
    await assert.rejects(packagePages(source, target), /ENOENT/);
    await writeFile(path.join(source, 'vendor/ort-wasm-simd-threaded.jsep.wasm'), 'fixture');
    await writeFile(path.join(source, 'index.html'), '<script src="/app.js"></script>');
    await assert.rejects(packagePages(source, target), /relative asset URL/);
    await writeFile(path.join(source, 'index.html'), '<script src="./missing.js"></script>');
    await assert.rejects(packagePages(source, target), /Unpackaged asset/);
    await writeFile(path.join(source, 'index.html'), 'fixture');
    cases.pop();
    await writeFile(path.join(source, 'model/reference.json'), JSON.stringify({cases}));
    await assert.rejects(packagePages(source, target));
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});
