import {RecallRuntime} from './runtime.js';
import {parseTokens, tensorEstimates} from './core.js';

const $ = id => document.getElementById(id);
const runtime = new RecallRuntime();
let manifest, lastResult, lastTokens;
let ready = false;
function busy(value) {
  for (const id of ['length', 'tokens', 'run', 'verify']) $(id).disabled = value || !ready;
  $('head').disabled = value || !lastResult;
}
function status(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
function clearResult() {
  lastResult = null;
  $('result').hidden = true;
  $('heatmap').replaceChildren();
  $('head').disabled = true;
}
function showMemory() {
  const estimate = tensorEstimates(manifest, Number($('length').value));
  const entry = manifest.models.find(entry => entry.length === Number($('length').value));
  $('memory').textContent = `ONNX file: ${entry.bytes.toLocaleString()} bytes. Float32 parameter tensor storage: ${estimate.parameters.toLocaleString()} bytes. One attention output estimate: ${estimate.attention.toLocaleString()} bytes (heads × length² × 4). Input: ${estimate.input} bytes; logits: ${estimate.logits} bytes. These are tensor/file sizes, not GPU allocation measurements.`;
}
async function loadLength() {
  clearResult();
  const length = Number($('length').value);
  status(`Loading length ${length} on WebGPU…`);
  await runtime.load(length);
  $('initialization').textContent = `Model fetch + session initialization: ${runtime.loadMs.toFixed(2)} ms. Two warmup runs: ${runtime.warmupMs.toFixed(2)} ms. Reloading a length replaces the previous session.`;
  showMemory();
  status('WebGPU ready · CPU compute fallback disabled.');
}
function renderAttention() {
  if (!lastResult) return;
  const length = lastTokens.length;
  const head = Number($('head').value);
  const table = document.createElement('table');
  const caption = table.createCaption();
  caption.textContent = `Head ${head} · rows: queries ↓ · columns: keys → · position: token ID`;
  const row = table.createTHead().insertRow();
  const corner = document.createElement('th');
  corner.textContent = 'Query ↓ / Key →';
  row.append(corner);
  lastTokens.forEach((token, i) => {
    const th = document.createElement('th');
    th.scope = 'col'; th.textContent = `${i}: ${token}`; row.append(th);
  });
  const body = table.createTBody();
  lastTokens.forEach((token, query) => {
    const tr = body.insertRow();
    const th = document.createElement('th');
    th.scope = 'row'; th.textContent = `${query}: ${token}`; tr.append(th);
    if (query === length - 1) tr.className = 'final-query';
    for (let key = 0; key < length; key++) {
      const weight = lastResult.attention[(head * length + query) * length + key];
      const cell = tr.insertCell();
      cell.textContent = weight.toFixed(2);
      cell.title = `Head ${head}, query ${query} (token ${token}), key ${key} (token ${lastTokens[key]}): ${weight.toPrecision(7)}`;
      cell.style.backgroundColor = `rgb(${Math.round(242 - 222 * weight)}, ${Math.round(248 - 164 * weight)}, ${Math.round(255 - 100 * weight)})`;
      cell.style.color = weight > 0.55 ? '#fff' : '#123';
    }
  });
  $('heatmap').replaceChildren(table);
}
$('head').addEventListener('change', renderAttention);
$('tokens').addEventListener('input', clearResult);
$('length').addEventListener('change', async () => {
  busy(true);
  $('tokens').value = Array.from({length: Number($('length').value)}, (_, i) => i % manifest.model_settings.classes).join(' ');
  try { await loadLength(); }
  catch (error) { status(error.message, true); }
  finally { busy(false); }
});
$('input-form').addEventListener('submit', async event => {
  event.preventDefault();
  busy(true);
  clearResult();
  try {
    const tokens = parseTokens($('tokens').value, Number($('length').value), manifest.model_settings.classes);
    if (!runtime.session) await loadLength();
    lastResult = await runtime.infer(tokens);
    lastTokens = tokens;
    $('prediction').textContent = `Predicted: ${manifest.vocabulary[lastResult.prediction]} · First input token: ${tokens[0]} · ${lastResult.prediction === tokens[0] ? 'Correct' : 'Incorrect'}`;
    $('logits').textContent = `Logits (class order ${manifest.vocabulary.join(', ')}): ${lastResult.logits.map(value => value.toFixed(4)).join(', ')}`;
    $('metrics').textContent = `WebGPU run + readback wall time: ${lastResult.wallMs.toFixed(2)} ms (one warmed sample).`;
    $('result').hidden = false;
    renderAttention();
    status('Completed locally with WebGPU · CPU compute fallback disabled.');
  } catch (error) { status(error.message, true); }
  finally { busy(false); }
});
$('verify').addEventListener('click', async () => {
  busy(true);
  clearResult();
  status('Comparing every exported length against saved PyTorch outputs…');
  $('verification').textContent = '';
  try {
    const report = await runtime.verify();
    $('verification').textContent = JSON.stringify(report, null, 2);
    await loadLength();
    status(`PASS: ${report.cases.length} PyTorch fixtures agree on WebGPU (logits, predictions and attention).`);
  } catch (error) {
    $('verification').textContent = `FAIL: ${error.message}`;
    status(error.message, true);
    // Verification may have switched sessions; restore the selected length.
    try { await loadLength(); status(error.message, true); }
    catch (restoreError) { status(restoreError.message, true); }
  } finally { busy(false); }
});
try {
  manifest = await runtime.initialize();
  for (const entry of manifest.models) $('length').add(new Option(String(entry.length), String(entry.length)));
  for (let i = 0; i < manifest.model_settings.heads; i++) $('head').add(new Option(`Head ${i}`, String(i)));
  $('model-info').textContent = `Checkpoint ${manifest.checkpoint_sha256.slice(0, 12)}… · ${manifest.model_settings.width} features · ${manifest.model_settings.heads} heads · trained at length ${manifest.training_config.length}, ${manifest.training_config.steps} steps. Other lengths may generalize poorly.`;
  $('token-help').textContent = `Valid IDs: ${manifest.vocabulary.join(', ')}. No padding, masks or text tokenization.`;
  $('tokens').value = Array.from({length: Number($('length').value)}, (_, i) => i % manifest.model_settings.classes).join(' ');
  await loadLength();
  ready = true;
  busy(false);
} catch (error) {
  status(`${error.message} See web/README.md for setup and export instructions.`, true);
}
