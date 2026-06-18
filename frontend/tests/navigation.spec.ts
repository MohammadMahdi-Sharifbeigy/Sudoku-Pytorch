import { expect, test } from "@playwright/test";

test("all nav links resolve without 404", async ({ page }) => {
  for (const path of ["/", "/solve", "/train", "/optimize"]) {
    await page.goto(path);
    await expect(page.getByText("This square has no number.")).toHaveCount(0);
  }
});
