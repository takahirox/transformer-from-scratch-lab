import {copyFile, mkdir, rm} from 'node:fs/promises';

const source = new URL('../node_modules/onnxruntime-web/', import.meta.url);
const target = new URL('../vendor/', import.meta.url);
await rm(target, {recursive: true, force: true});
await mkdir(target, {recursive: true});
for (const name of ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm']) {
  await copyFile(new URL(`dist/${name}`, source), new URL(name, target));
}
console.log('Prepared local ONNX Runtime Web 1.22.0 assets in web/vendor.');
