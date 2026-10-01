// Keep browser tests executable under the same script CSP as the application.
const moduleUrl = document.querySelector('script[data-test]').dataset.test;
try {
    const { run } = await import(new URL(moduleUrl, import.meta.url).href);
    window.altdReport = await run(new URL('../../', import.meta.url).href);
} catch (error) {
    window.altdReport = { ok: false, error: String(error.stack || error) };
}
document.querySelector('#output').textContent = JSON.stringify(window.altdReport, null, 2);
