import { expect, test } from "@playwright/test";

async function mockOptimizeApi(page: import("@playwright/test").Page) {
  await page.route("**/api/model/info", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        exists: true,
        size_mb: 1.234,
        created_at: "2026-06-18T08:00:00Z",
        total_params: 123456,
        trainable_params: 123456,
      }),
    });
  });

  await page.route("**/api/optimize", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        success: true,
        results: [
          { format: "pt", size_mb: 1.234, latency_ms: 2.8, path: "models/best_model.pt" },
          { format: "ts", size_mb: 1.12, latency_ms: 1.9, path: "models/best_model.ts" },
          { format: "onnx", size_mb: 0.98, latency_ms: 1.4, path: "models/best_model.onnx" },
        ],
      }),
    });
  });
}

test("run benchmark and verify results table", async ({ page }) => {
  await mockOptimizeApi(page);
  await page.goto("/optimize");

  await expect(page.getByText("Ready to optimize")).toBeVisible();
  await page.getByRole("button", { name: "Run Optimization & Benchmark" }).click();

  await expect(page.getByText("Benchmark complete")).toBeVisible();
  await expect(page.getByRole("cell", { name: "ONNX" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "Fastest" })).toBeVisible();
});
