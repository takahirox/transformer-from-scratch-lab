import {copyFile, mkdir, rm, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';

const source = new URL('../node_modules/onnxruntime-web/', import.meta.url);
const target = new URL('../vendor/', import.meta.url);
await rm(target, {recursive: true, force: true});
await mkdir(target, {recursive: true});
for (const name of ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm']) {
  await copyFile(new URL(`dist/${name}`, source), new URL(name, target));
}
// The npm package omits the upstream third-party notices. Ship the exact
// version's notices alongside the runtime, checking integrity before packaging.
const response = await fetch('https://raw.githubusercontent.com/microsoft/onnxruntime/f217402897f40ebba457e2421bc0a4702771968e/ThirdPartyNotices.txt');
if (!response.ok) throw new Error(`Runtime notices download failed: HTTP ${response.status}`);
const notices = Buffer.from(await response.arrayBuffer());
if (createHash('sha256').update(notices).digest('hex') !==
    'e9e90971a8e75a9a8ac0c6412e29c1202d079998389915aa485f46c816c3b4cc') {
  throw new Error('Runtime notices integrity mismatch');
}
await writeFile(new URL('ThirdPartyNotices.txt', target), notices);
console.log('Prepared local ONNX Runtime Web 1.22.0 assets in web/vendor.');
