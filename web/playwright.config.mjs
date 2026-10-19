import {defineConfig} from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.mjs',
  workers: 1,
  timeout: 120_000,
  use: {baseURL: 'http://127.0.0.1:18765', channel: 'chrome', headless: true},
  webServer: {
    command: 'python3 -m http.server 18765 --bind 127.0.0.1 --directory .',
    url: 'http://127.0.0.1:18765',
    reuseExistingServer: false,
  },
});
