import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
const app = process.env.APP_ROOT;
if (!app || process.platform !== "linux")
  throw new Error(
    "Set APP_ROOT to the built webapp worktree; this check requires Linux.",
  );
const require = createRequire(`${app}/package.json`);
const { chromium } = require("@playwright/test");
const server = spawn("node", ["scripts/serve-test.mjs"], {
  cwd: app,
  stdio: ["ignore", "ignore", "inherit"],
});
let browser;
try {
  for (let i = 0; i < 100; i++) {
    try {
      if ((await fetch("http://127.0.0.1:4173")).ok) break;
    } catch {}
    await new Promise((r) => setTimeout(r, 100));
  }
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
  });
  await page.addInitScript(() => {
    const Native = Worker;
    window.Worker = class extends Native {
      postMessage(command) {
        if (command.type === "start") command.threadCount = 4;
        super.postMessage(command);
      }
    };
  });
  const cdp = await browser.newBrowserCDPSession();
  const pageCdp = await page.context().newCDPSession(page);
  await pageCdp.send("HeapProfiler.enable");
  let peakTotalRssBytes = 0;
  const byteValue = (text, key) =>
    Number(text.match(new RegExp("^" + key + ":\\s+(\\d+)", "m"))?.[1] ?? 0) *
    1024;
  const processMemory = async () => {
    const { processInfo } = await cdp.send("SystemInfo.getProcessInfo");
    const result = [];
    for (const process of processInfo) {
      try {
        const status = await readFile(`/proc/${process.id}/status`, "utf8");
        const smaps = await readFile(`/proc/${process.id}/smaps`, "utf8");
        const largeAnonymousMappings = smaps
          .split(/(?=^[0-9a-f]+-[0-9a-f]+ )/m)
          .flatMap((block) => {
            const header = block.split("\n")[0];
            const sizeBytes = byteValue(block, "Size");
            if (sizeBytes < 1024 ** 3 || /\//.test(header)) return [];
            return [{ sizeBytes, rssBytes: byteValue(block, "Rss") }];
          });
        result.push({
          type: process.type,
          rssBytes: byteValue(status, "VmRSS"),
          virtualBytes: byteValue(status, "VmSize"),
          largeAnonymousMappings,
        });
      } catch {}
    }
    peakTotalRssBytes = Math.max(
      peakTotalRssBytes,
      result.reduce((sum, p) => sum + p.rssBytes, 0),
    );
    return result;
  };
  const sampling = setInterval(() => {
    void processMemory().catch(() => {});
  }, 100);
  sampling.unref();
  await page.goto("http://127.0.0.1:4173/");
  await page
    .locator("#track-files")
    .setInputFiles(`${app}/tests/fixtures/Test Loop.track`);
  await page.locator("#count-tracks").waitFor();
  await page.goto("http://127.0.0.1:4173/#new");
  for (const [key, value] of Object.entries({
    population: 4096,
    ticks: 12000,
    generations: 20,
  })) {
    await page.locator(`[data-training="${key}"]`).fill(String(value));
    await page.locator(`[data-training="${key}"]`).dispatchEvent("change");
  }
  await page.locator('[data-training="eliminateOnWall"]').uncheck();
  await page.locator('[data-training="eliminateWhenIdle"]').uncheck();
  const samples = [];
  async function dump(label) {
    if (!label.startsWith("live"))
      await pageCdp.send("HeapProfiler.collectGarbage");
    const processes = await processMemory();
    samples.push({ label, processes });
  }
  const count = async () =>
    (await cdp.send("Target.getTargets")).targetInfos.filter(
      (t) => t.type === "worker",
    ).length;
  await dump("baseline");
  for (let cycle = 0; cycle < 4; cycle++) {
    if (cycle) {
      await page.goto("http://127.0.0.1:4173/#new");
      for (const [key, value] of Object.entries({
        population: 4096,
        ticks: 12000,
        generations: 20,
      })) {
        await page.locator(`[data-training="${key}"]`).fill(String(value));
        await page.locator(`[data-training="${key}"]`).dispatchEvent("change");
      }
      await page.locator('[data-training="eliminateOnWall"]').uncheck();
      await page.locator('[data-training="eliminateWhenIdle"]').uncheck();
    }
    await page.locator("#start").click();
    await page
      .locator("#live-figures")
      .filter({ hasText: "CPU, 4 threads" })
      .waitFor({ timeout: 30000 });
    await page.locator("#toggle-run").click();
    await page
      .locator("#run-state")
      .filter({ hasText: /^Paused/ })
      .waitFor();
    await dump(`live-${cycle}`);
    page.once("dialog", (d) => d.accept());
    await page.locator("#delete-run").click();
    for (let i = 0; i < 100 && (await count()); i++)
      await new Promise((r) => setTimeout(r, 50));
    if (await count()) throw new Error("surviving workers");
    await dump(`deleted-${cycle}`);
  }
  clearInterval(sampling);
  console.log(
    JSON.stringify(
      {
        browser: browser.version(),
        population: 4096,
        threads: 4,
        cycles: 4,
        samplingIntervalMs: 100,
        peakTotalRssBytes,
        samples,
      },
      null,
      2,
    ),
  );
} finally {
  if (browser) await browser.close();
  server.kill();
}
