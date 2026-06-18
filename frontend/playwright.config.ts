import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  expect: { timeout: 8_000 },
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:8080",
    trace: "on-first-retry",
  },
  webServer: {
    command: "node tests/start-next-server.mjs",
    url: "http://localhost:8080",
    reuseExistingServer: true,
    stdout: "ignore",
    stderr: "ignore",
    timeout: 120_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
