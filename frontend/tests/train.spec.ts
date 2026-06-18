import { expect, test } from "@playwright/test";

async function mockTrainingStream(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    class MockEventSource extends EventTarget {
      url: string;
      readyState = 1;
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;

      constructor(url: string) {
        super();
        this.url = url;
        window.setTimeout(() => {
          this.onopen?.(new Event("open"));
          [
            { type: "epoch", epoch: 1, epochs: 2, train_loss: 0.7, val_loss: 0.6, train_acc: 72, val_acc: 75 },
            { type: "epoch", epoch: 2, epochs: 2, train_loss: 0.3, val_loss: 0.25, train_acc: 91, val_acc: 92 },
            { type: "best_model", epoch: 2, val_loss: 0.25 },
            { type: "test_results", test_loss: 0.22, test_acc: 93.5, y_true: [0, 1, 2, 3], y_pred: [0, 1, 2, 3] },
            { type: "complete", model_path: "models/best_model.pt" },
          ].forEach((payload, index) => {
            window.setTimeout(() => {
              this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(payload) }));
            }, 150 * (index + 1));
          });
        }, 50);
      }

      close() {
        this.readyState = 2;
      }
    }

    Object.defineProperty(window, "EventSource", {
      value: MockEventSource,
      writable: true,
    });
  });
}

test("start training and verify SSE chart updates", async ({ page }) => {
  await mockTrainingStream(page);
  await page.goto("/train");

  await page.getByRole("button", { name: "Start Training" }).click();

  await expect(page.getByText("Epoch 2")).toBeVisible();
  await expect(page.getByText("Loss curve")).toBeVisible();
  await expect(page.getByText("Training complete", { exact: true })).toBeVisible();
  await expect(page.getByText("Final results")).toBeVisible();
});
