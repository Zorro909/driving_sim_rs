// The coordinator must live off the page, including during pool startup.
const worker = new Worker(new URL('./threads.worker.js', import.meta.url), { type: 'module' });
const timeout = setTimeout(() => {
    worker.terminate();
    window.altdReport = { ok: false, error: 'Threaded worker did not finish within five minutes' };
}, 300000);
const latencies = [];
const ping = setInterval(() => worker.postMessage({ type: "ping", sent: performance.timeOrigin + performance.now() }), 10);
worker.onmessage = ({ data }) => {
    if (data.type === "pong") { latencies.push(performance.timeOrigin + performance.now() - data.sent); return; }
    clearInterval(ping);
    data.messageLatencyMs = latencies.length ? Math.max(...latencies) : null;
    clearTimeout(timeout);
    worker.terminate();
    window.altdReport = data;
};
worker.onerror = (event) => {
    clearInterval(ping);
    clearTimeout(timeout);
    worker.terminate();
    window.altdReport = { ok: false, error: event.message };
};
worker.postMessage(window.altdTestOptions);
