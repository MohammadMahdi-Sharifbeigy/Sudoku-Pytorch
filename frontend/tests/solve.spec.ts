import { expect, test } from "@playwright/test";

const pngBase64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=";

async function mockSolveApi(page: import("@playwright/test").Page) {
  await page.route("**/api/solve", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        success: true,
        solved: true,
        original_grid: [
          [5, 3, 0, 0, 7, 0, 0, 0, 0],
          [6, 0, 0, 1, 9, 5, 0, 0, 0],
          [0, 9, 8, 0, 0, 0, 0, 6, 0],
          [8, 0, 0, 0, 6, 0, 0, 0, 3],
          [4, 0, 0, 8, 0, 3, 0, 0, 1],
          [7, 0, 0, 0, 2, 0, 0, 0, 6],
          [0, 6, 0, 0, 0, 0, 2, 8, 0],
          [0, 0, 0, 4, 1, 9, 0, 0, 5],
          [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ],
        solved_board: [
          [5, 3, 4, 6, 7, 8, 9, 1, 2],
          [6, 7, 2, 1, 9, 5, 3, 4, 8],
          [1, 9, 8, 3, 4, 2, 5, 6, 7],
          [8, 5, 9, 7, 6, 1, 4, 2, 3],
          [4, 2, 6, 8, 5, 3, 7, 9, 1],
          [7, 1, 3, 9, 2, 4, 8, 5, 6],
          [9, 6, 1, 5, 3, 7, 2, 8, 4],
          [2, 8, 7, 4, 1, 9, 6, 3, 5],
          [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ],
        per_cell: Array.from({ length: 81 }, (_, index) => ({
          index,
          row: Math.floor(index / 9),
          col: index % 9,
          has_digit: index % 5 === 0,
          label: (index % 9) + 1,
          confidence: index % 10 === 0 ? 0.42 : 0.91,
          cell_image_b64: pngBase64,
        })),
        confidence_stats: {
          digit_count: 20,
          empty_count: 61,
          avg_confidence: 0.88,
          min_confidence: 0.42,
          low_confidence_count: 2,
        },
        solution_image_b64: pngBase64,
        board_image_b64: pngBase64,
      }),
    });
  });
}

async function mockModelLists(page: import("@playwright/test").Page) {
  await page.route("**/api/models/cnn", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        models: [
          {
            id: "cnn-best",
            filename: "best_model.pt",
            size_mb: 1.2,
            created_at: "2026-06-20T00:00:00Z",
            is_default: true,
          },
        ],
      }),
    });
  });

  await page.route("**/api/models/yolo", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        models: [
          {
            id: "pose-best",
            filename: "sudoku_pose_best.pt",
            size_mb: 21.5,
            created_at: "2026-06-20T00:00:00Z",
            is_default: true,
          },
        ],
      }),
    });
  });
}

test("detector toggle and model selects react to detector choice", async ({ page }) => {
  await mockModelLists(page);
  await mockSolveApi(page);
  await page.goto("/solve");

  // Detector toggle (Classical / YOLO) is visible.
  const classical = page.getByRole("radio", { name: "Classical" });
  const yolo = page.getByRole("radio", { name: "YOLO" });
  await expect(classical).toBeVisible();
  await expect(yolo).toBeVisible();

  // CNN model select is always visible.
  await expect(page.getByText("Digit model (CNN)")).toBeVisible();

  // Default detector is YOLO → pose model select is visible.
  await expect(page.getByText("Grid model (pose)")).toBeVisible();

  // Switching to Classical hides the pose select.
  await classical.click();
  await expect(page.getByText("Grid model (pose)")).toHaveCount(0);

  // Switching back to YOLO reveals the pose select again.
  await yolo.click();
  await expect(page.getByText("Grid model (pose)")).toBeVisible();
});

test("upload image and verify solved grid appears", async ({ page }) => {
  await mockModelLists(page);
  await mockSolveApi(page);
  await page.goto("/solve");

  const file = Buffer.from(
    pngBase64,
    "base64",
  );

  await page.locator('input[type="file"]').setInputFiles({
    name: "sudoku.png",
    mimeType: "image/png",
    buffer: file,
  });
  await page.getByRole("button", { name: "Solve uploaded sudoku" }).click();

  await expect(page.getByText("Puzzle solved").first()).toBeVisible();
  await expect(page.getByRole("tab", { name: "Solution" })).toBeVisible();
});
