// Pure input/verification helpers shared by the UI and tests.
export function parseTokens(text, length, classes) {
  const parts = text.trim().split(/\s+/);
  if (parts.length !== length) throw new Error(`Enter exactly ${length} tokens; received ${text.trim() ? parts.length : 0}.`);
  if (parts.some(token => !/^\d+$/.test(token) || !Number.isSafeInteger(Number(token)) || Number(token) >= classes)) {
    throw new Error(`Tokens must be integer IDs from 0 to ${classes - 1}.`);
  }
  return parts.map(Number);
}

export function argmax(values) {
  if (!values.length || values.some(value => !Number.isFinite(value))) throw new Error('Invalid model output.');
  return values.reduce((best, value, index) => value > values[best] ? index : best, 0);
}

export function compare(actual, expected, {atol, rtol}) {
  if (actual.length !== expected.length) throw new Error('Output shape differs from the reference.');
  let maxAbsError = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(error) || error > atol + rtol * Math.abs(expected[i])) {
      throw new Error(`Numerical mismatch at ${i}: ${actual[i]} vs ${expected[i]}.`);
    }
    maxAbsError = Math.max(maxAbsError, error);
  }
  return maxAbsError;
}

export function tensorEstimates(manifest, length) {
  return {
    parameters: manifest.parameter_bytes,
    attention: manifest.model_settings.heads * length * length * 4,
    logits: manifest.model_settings.classes * 4,
    input: length * 4,
  };
}
