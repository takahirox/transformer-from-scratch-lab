import {test, expect} from '@playwright/test';
import {readFile, writeFile} from 'node:fs/promises';

test('unavailable WebGPU has a clear error and no enabled inference controls', async ({page}) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', {value: undefined}));
  await page.goto('/');
  await expect(page.locator('#status')).toContainText('WebGPU is unavailable');
  await expect(page.locator('#run')).toBeDisabled();
  await expect(page.locator('#result')).toBeHidden();
});

test('unsupported WebGPU operators fail without CPU compute fallback', async ({page}) => {
  const fixture = JSON.parse(await readFile(new URL('./unsupported-model.json', import.meta.url), 'utf8'));
  await page.route('**/model/*.onnx', route => route.fulfill({
    status: 200, contentType: 'application/octet-stream', body: Buffer.from(fixture.base64, 'base64'),
  }));
  await page.goto('/');
  const hasAdapter = await page.evaluate(async () => !!await navigator.gpu?.requestAdapter());
  test.skip(!hasAdapter, 'No WebGPU adapter: cannot verify operator rejection.');
  await expect(page.locator('#status')).toContainText('WebGPU model initialization failed', {timeout: 60_000});
  // Check the runtime diagnostic itself, in addition to the app's status prefix.
  await expect(page.locator('#status')).toContainText('fallback to CPU EP has been explicitly disabled');
  await expect(page.locator('#run')).toBeDisabled();
});

test('WebGPU agrees with PyTorch and supports local input, lengths and attention heads', async ({page}, testInfo) => {
  const errors = [];
  const requests = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => {
    requests.push(route.request().url());
    if (new URL(route.request().url()).origin !== 'http://127.0.0.1:18765') return route.abort();
    return route.continue();
  });
  await page.goto('/');
  const capability = await page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter();
    return {secureContext: isSecureContext, navigatorGPU: !!navigator.gpu, adapterAvailable: !!adapter};
  });
  await writeFile(testInfo.outputPath('capability.json'), JSON.stringify(capability, null, 2));
  test.skip(!capability.adapterAvailable, `Verified WebGPU limitation: ${JSON.stringify(capability)}`);
  await expect.poll(async () => await page.locator('#run').isEnabled() ||
    await page.locator('#status').getAttribute('class') === 'error', {timeout: 60_000}).toBe(true);
  await expect(page.locator('#status')).toContainText('WebGPU ready');
  await page.locator('#verify').click();
  await expect(page.locator('#verification')).not.toBeEmpty({timeout: 90_000});
  await expect(page.locator('#status')).toContainText('PASS:');
  const report = JSON.parse(await page.locator('#verification').textContent());
  expect(report.provider).toBe('webgpu');
  expect(report.cpuFallbackDisabled).toBe(true);
  const manifest = await (await page.request.get('/model/manifest.json')).json();
  expect(report.cases).toHaveLength(manifest.models.length * 3);
  await page.locator('#tokens').fill('999');
  await page.locator('#run').click();
  await expect(page.locator('#status')).toContainText('Enter exactly');
  for (const entry of manifest.models) {
    await page.locator('#length').selectOption(String(entry.length));
    await expect(page.locator('#run')).toBeEnabled();
    await page.locator('#tokens').fill(['999', ...Array(entry.length - 1).fill('0')].join(' '));
    await page.locator('#run').click();
    await expect(page.locator('#status')).toContainText('Tokens must be integer IDs');
    await page.locator('#tokens').fill(Array.from({length: entry.length}, (_, i) => i % manifest.model_settings.classes).join(' '));
    await page.locator('#run').click();
    await expect(page.locator('#status')).toContainText('Completed locally with WebGPU');
    await expect(page.locator('#prediction')).toContainText('Predicted:');
    await expect(page.locator('#heatmap tbody tr')).toHaveCount(entry.length);
    await page.locator('#head').selectOption(String(manifest.model_settings.heads - 1));
    await expect(page.locator('#heatmap caption')).toContainText(`Head ${manifest.model_settings.heads - 1}`);
  }
  expect(errors).toEqual([]);
  expect(requests.every(url => new URL(url).origin === 'http://127.0.0.1:18765')).toBe(true);
  report.requestOrigins = [...new Set(requests.map(url => new URL(url).origin))];
  await writeFile(testInfo.outputPath('webgpu.json'), JSON.stringify(report, null, 2) + '\n');
  await page.screenshot({path: testInfo.outputPath('demo.png'), fullPage: true});
});
