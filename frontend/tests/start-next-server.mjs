import { spawn, spawnSync } from "node:child_process";

const child = spawn(process.execPath, ["node_modules/next/dist/bin/next", "dev", "-p", "8080"], {
  cwd: process.cwd(),
  env: process.env,
  stdio: "inherit",
  windowsHide: true,
});

let shuttingDown = false;

function killChildTree() {
  if (!child.pid) return;

  spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], {
    stdio: "ignore",
    windowsHide: true,
  });
}

function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  killChildTree();
  process.exit(signal === "SIGTERM" || signal === "SIGINT" ? 0 : 1);
}

child.on("exit", (code, signal) => {
  if (!shuttingDown) {
    process.exit(code ?? (signal ? 1 : 0));
  }
});

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
process.on("exit", () => {
  if (!shuttingDown) {
    killChildTree();
  }
});
