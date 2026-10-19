import assert from 'node:assert/strict';
import {test} from 'node:test';
import {argmax, compare, parseTokens, tensorEstimates} from '../core.js';

test('only exact-length sequences of in-vocabulary integer IDs are accepted', () => {
  assert.deepEqual(parseTokens(' 0  3\n2 ', 3, 4), [0, 3, 2]);
  for (const input of ['', '0 1', '0 1 2 3', '-1 0 1', '4 0 1', '1.5 0 1', '1e0 0 1', 'NaN 0 1']) {
    assert.throws(() => parseTokens(input, 3, 4));
  }
});

test('numeric verification rejects drift, shape mismatch and nonfinite values', () => {
  const tolerance = {atol: 1e-4, rtol: 1e-3};
  assert.ok(compare([1.0001, 0.00001], [1, 0], tolerance) < 0.0002);
  assert.throws(() => compare([1.01], [1], tolerance));
  assert.throws(() => compare([], [1], tolerance));
  assert.throws(() => compare([NaN], [1], tolerance));
  assert.throws(() => compare([Infinity], [1], tolerance));
  assert.equal(argmax([-3, 5, 2]), 1);
  assert.throws(() => argmax([NaN]));
});

test('memory estimates use batch 1 and float32 tensor sizes', () => {
  assert.deepEqual(tensorEstimates({parameter_bytes: 800, model_settings: {heads: 4, classes: 8}}, 16),
    {parameters: 800, attention: 4096, logits: 32, input: 64});
});
