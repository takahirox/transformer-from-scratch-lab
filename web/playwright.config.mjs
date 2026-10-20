import {defineConfig} from '@playwright/test';

const pages = process.env.PAGES_TEST === '1';
const origin = 'http://127.0.0.1:18765';
const baseURL = `${origin}${pages ? '/transformer-from-scratch-lab/' : '/'}`;

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.mjs',
  workers: 1,
  timeout: 120_000,
  reporter: [['list'], ['junit', {outputFile: 'test-results/results.xml'}]],
  use: {baseURL, channel: 'chrome', headless: true},
  webServer: {
    command: `python3 -m http.server 18765 --bind 127.0.0.1 --directory ${pages ? '../results/pages' : '.'}`,
    url: baseURL,
    reuseExistingServer: false,
  },
});
