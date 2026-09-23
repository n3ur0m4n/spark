/** Force collection between jobs, not while a WeakRef is kept alive by deref. */
export async function checkNativeRetention(page) {
  await page.waitForFunction(() => window.retentionDiagnostic, null, {
    timeout: 120000,
  });
  const cdp = await page.context().newCDPSession(page);
  try {
    for (let i = 0; i < 3; i++) {
      await page.evaluate(
        () => new Promise((resolve) => setTimeout(resolve, 0)),
      );
      await cdp.send("HeapProfiler.collectGarbage");
    }
    return await page.evaluate(() => window.finishRetention());
  } finally {
    await cdp.detach();
  }
}
