import {test, expect} from '@playwright/test';
import {createHash} from 'node:crypto';

test('packaged site serves all assets with correct paths and MIME types', async ({page, request, baseURL}) => {
  test.skip(process.env.PAGES_TEST !== '1', 'Run test:pages against the assembled artifact.');
  const manifestResponse = await request.get('model/manifest.json');
  expect(manifestResponse.ok()).toBe(true);
  const manifest = await manifestResponse.json();
  const files = ['index.html', 'style.css', 'app.js', 'core.js', 'runtime.js',
    'THIRD_PARTY_NOTICES.txt', 'vendor/ThirdPartyNotices.txt',
    'vendor/ort.webgpu.min.mjs', 'vendor/ort-wasm-simd-threaded.jsep.mjs',
    'vendor/ort-wasm-simd-threaded.jsep.wasm', `model/${manifest.reference}`,
    ...manifest.models.map(entry => `model/${entry.file}`)];
  for (const file of files) {
    const response = await request.get(file);
    expect(response.ok(), file).toBe(true);
    expect(response.url()).toBe(`${baseURL}${file}`);
    const body = await response.body();
    expect(body.length, file).toBeGreaterThan(0);
    if (/\.m?js$/.test(file)) expect(response.headers()['content-type']).toMatch(/javascript/);
    if (file.endsWith('.wasm')) expect(response.headers()['content-type']).toBe('application/wasm');
    const entry = manifest.models.find(entry => file === `model/${entry.file}`);
    if (entry) {
      expect(body.length).toBe(entry.bytes);
      expect(createHash('sha256').update(body).digest('hex')).toBe(entry.sha256);
    }
  }
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('requestfailed', req => errors.push(req.url()));
  page.on('response', response => {if (!response.ok()) errors.push(response.url());});
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', {value: undefined}));
  await page.goto('./');
  await expect(page.locator('#status')).toContainText('WebGPU is unavailable');
  await expect(page.locator('#run')).toBeDisabled();
  expect(errors).toEqual([]);
});
