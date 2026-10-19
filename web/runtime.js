import {argmax, compare} from './core.js';

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}. Train/export the model first.`);
  return response.json();
}

export class RecallRuntime {
  async initialize() {
    if (!globalThis.isSecureContext || !navigator.gpu) {
      throw new Error('WebGPU is unavailable. Use a supported desktop browser on HTTPS or localhost with GPU acceleration enabled.');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter is available. Check browser/hardware support and GPU acceleration.');
    this.adapterInfo = adapter.info ? {
      vendor: adapter.info.vendor, architecture: adapter.info.architecture,
      device: adapter.info.device, description: adapter.info.description,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    } : {};
    this.ort = await import('./vendor/ort.webgpu.min.mjs');
    this.ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
    this.ort.env.wasm.numThreads = 1;
    this.ort.env.wasm.proxy = false;
    this.ort.env.webgpu.adapter = adapter;
    this.manifest = await getJSON(new URL('./model/manifest.json', import.meta.url));
    if (this.manifest.format_version !== 1) throw new Error('Unsupported export format. Re-export the checkpoint.');
    return this.manifest;
  }

  async load(length) {
    const entry = this.manifest.models.find(model => model.length === length);
    if (!entry) throw new Error('This sequence length was not exported.');
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    this.length = length;
    const start = performance.now();
    try {
      this.session = await this.ort.InferenceSession.create(new URL(`./model/${entry.file}`, import.meta.url).href, {
        executionProviders: ['webgpu'],
        // Fail on unsupported compute nodes instead of accepting WASM/CPU fallback.
        extra: {session: {disable_cpu_ep_fallback: '1'}},
        graphOptimizationLevel: 'all',
      });
      if (!this.device) {
        this.device = this.ort.env.webgpu.device;
        this.device.lost.then(info => { this.lost = info.message || 'WebGPU device lost; reload the page.'; });
      }
      this.loadMs = performance.now() - start;
      const warmupStart = performance.now();
      for (let i = 0; i < 2; i++) await this.infer(Array(length).fill(0));
      this.warmupMs = performance.now() - warmupStart;
    } catch (error) {
      if (this.session) await this.session.release();
      this.session = null;
      throw new Error(`WebGPU model initialization failed (unsupported operator/device or missing model). CPU fallback is disabled. ${error.message ?? error}`);
    }
  }

  async infer(tokens) {
    if (this.lost) throw new Error(this.lost);
    if (!this.session) throw new Error('Load a WebGPU model first.');
    if (tokens.length !== this.length || tokens.some(token => !Number.isInteger(token) || token < 0 || token >= this.manifest.model_settings.classes)) {
      throw new Error('Invalid token IDs or sequence length.');
    }
    const input = new this.ort.Tensor('int32', Int32Array.from(tokens), [1, tokens.length]);
    let outputs;
    try {
      const start = performance.now();
      outputs = await this.session.run({tokens: input});
      // CPU outputs force readback; additionally wait for submitted GPU work.
      await this.device.queue.onSubmittedWorkDone();
      const wallMs = performance.now() - start;
      const logits = Array.from(outputs.logits.data);
      const attention = Array.from(outputs.attention.data);
      const {heads, classes} = this.manifest.model_settings;
      if (outputs.logits.dims.join() !== [1, classes].join() ||
          outputs.attention.dims.join() !== [1, heads, this.length, this.length].join() ||
          attention.some(value => !Number.isFinite(value))) throw new Error('Unexpected model output shape or values.');
      return {logits, attention, prediction: argmax(logits), wallMs};
    } finally {
      input.dispose();
      if (outputs) Object.values(outputs).forEach(tensor => tensor.dispose());
    }
  }

  async verify() {
    const reference = await getJSON(new URL(`./model/${this.manifest.reference}`, import.meta.url));
    const results = [];
    for (const entry of this.manifest.models) {
      await this.load(entry.length);
      for (const test of reference.cases.filter(test => test.length === entry.length)) {
        const actual = await this.infer(test.tokens);
        const logitsMaxAbsError = compare(actual.logits, test.logits, reference.tolerance);
        const attentionMaxAbsError = compare(actual.attention, test.attention, reference.tolerance);
        if (actual.prediction !== test.prediction) throw new Error('Prediction differs from PyTorch.');
        results.push({length: test.length, tokens: test.tokens, prediction: actual.prediction,
          logitsMaxAbsError, attentionMaxAbsError, wallMs: actual.wallMs});
      }
    }
    return {provider: 'webgpu', cpuFallbackDisabled: true, adapter: this.adapterInfo,
      userAgent: navigator.userAgent, tolerance: reference.tolerance, cases: results};
  }
}
