// Optional real-browser acceptance: Chrome CDP on 9222, frontend on 8000,
// isolated fixture backend on 8081. Never points saves at the normal 8080 store.
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
const targets = await (await fetch("http://127.0.0.1:9222/json")).json();
const ws = new WebSocket(
    targets.find((t) => t.type === "page").webSocketDebuggerUrl,
);
await new Promise((resolve) =>
    ws.addEventListener("open", resolve, { once: true }),
);
let sequence = 0;
const pending = new Map();
const sent = [];
let saveMode = "normal";
let failLoad = false;
const errors = [];
const call = (method, params = {}) =>
    new Promise((resolve, reject) => {
        const id = ++sequence;
        pending.set(id, { resolve, reject });
        ws.send(JSON.stringify({ id, method, params }));
    });
ws.addEventListener("message", async ({ data }) => {
    const message = JSON.parse(data);
    if (message.id) {
        const entry = pending.get(message.id);
        pending.delete(message.id);
        if (message.error)
            entry.reject(new Error(JSON.stringify(message.error)));
        else entry.resolve(message.result);
        return;
    }
    if (message.method === "Runtime.exceptionThrown")
        errors.push(message.params.exceptionDetails.text);
    if (message.method !== "Fetch.requestPaused") return;
    const { requestId, request } = message.params;
    if (request.url.endsWith("/annotation-config.js")) {
        await call("Fetch.fulfillRequest", {
            requestId,
            responseCode: 200,
            responseHeaders: [
                { name: "Content-Type", value: "application/javascript" },
            ],
            body: Buffer.from(
                'window.annotationConfig={provider:"server",baseUrl:"http://127.0.0.1:8081/api"};',
            ).toString("base64"),
        });
    } else if (
        request.url.endsWith("/annotations") &&
        request.method === "POST"
    ) {
        sent.push(JSON.parse(request.postData));
        if (saveMode === "offline")
            await call("Fetch.failRequest", {
                requestId,
                errorReason: "ConnectionFailed",
            });
        else if (saveMode === "badAck")
            await call("Fetch.fulfillRequest", {
                requestId,
                responseCode: 200,
                responseHeaders: [
                    { name: "Content-Type", value: "application/json" },
                    {
                        name: "Access-Control-Allow-Origin",
                        value: "http://127.0.0.1:8000",
                    },
                ],
                body: Buffer.from(
                    '{"submissionId":"wrong","savedCount":1}',
                ).toString("base64"),
            });
        else await call("Fetch.continueRequest", { requestId });
    } else if (failLoad && request.url.endsWith("/candidates")) {
        await call("Fetch.failRequest", {
            requestId,
            errorReason: "ConnectionFailed",
        });
    } else await call("Fetch.continueRequest", { requestId });
});
const evaluate = async (expression) => {
    const result = await call("Runtime.evaluate", {
        expression,
        awaitPromise: true,
        returnByValue: true,
    });
    if (result.exceptionDetails)
        throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
};
const until = async (expression) => {
    const end = Date.now() + 20000;
    while (Date.now() < end) {
        if (await evaluate(expression)) return;
        await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw new Error(
        `Timed out: ${expression}; status=${await evaluate('document.querySelector("#status")?.textContent')}`,
    );
};
const click = (selector) =>
    evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
try {
    await call("Runtime.enable");
    await call("Page.enable");
    await call("Fetch.enable", {
        patterns: [
            { urlPattern: "*annotation-config.js" },
            { urlPattern: "*:8081/api/*" },
        ],
    });
    await call("Page.navigate", {
        url: "http://127.0.0.1:8000/docs/annotate.html",
    });
    await until('document.querySelectorAll(".answer-options").length === 15');
    assert.equal(
        await evaluate('document.querySelectorAll("#loadRemaining").length'),
        1,
    );
    assert.equal(
        await evaluate('document.querySelector("#scientific").value'),
        "DENEBR",
    );
    assert.equal(
        await evaluate(
            'document.querySelectorAll("#frog-card select option").length',
        ),
        3,
    );
    assert.match(
        await evaluate('document.querySelector("legend").textContent'),
        /Dendropsophus ebraccatus/,
    );
    assert.equal(
        await evaluate('document.querySelectorAll(".results-list img").length'),
        0,
    );
    assert.equal(
        await evaluate('document.querySelector("#frog-card audio").controls'),
        true,
    );
    assert.equal(
        await evaluate(
            'document.querySelector(".results-list audio").controls',
        ),
        false,
    );
    await evaluate('document.querySelector("#frog-card audio").play()');
    await until('!document.querySelector("#frog-card audio").paused');
    await click(".results-list .audio-clip button");
    await until('!document.querySelector(".results-list audio").paused');
    assert.equal(
        await evaluate('document.querySelector("#frog-card audio").paused'),
        true,
    );
    const start = await evaluate(
        'document.querySelector(".results-list audio").currentTime',
    );
    assert.ok(start >= 50 && start < 55, `candidate start ${start}`);
    await until('document.querySelector(".results-list audio").paused');
    await click(".results-list .audio-clip button");
    await until('!document.querySelector(".results-list audio").paused');
    assert.ok(
        await evaluate(
            'document.querySelector(".results-list audio").currentTime < 51',
        ),
    );
    await click(".results-list .audio-clip button");
    for (const [i, label] of ["yes", "no", "unsure"].entries())
        await click(
            `.results-list > li:nth-child(${i + 1}) input[value="${label}"]`,
        );
    await click("#sendButton");
    await until(
        'document.querySelector("#status").textContent.startsWith("Annotations saved.")',
    );
    assert.deepEqual(
        sent[0].annotations.map((a) => a.label),
        ["yes", "no", "unsure"],
    );
    const savedIds = sent[0].annotations.map((a) => a.windowId);
    failLoad = true;
    await click("#loadRemaining");
    await until(
        'document.querySelector("#status").textContent.includes("Could not load remaining work")',
    );
    assert.match(
        await evaluate('document.querySelector("#status").textContent'),
        /were saved/,
    );
    assert.equal(
        await evaluate('document.querySelectorAll(".answer-options").length'),
        15,
    );
    failLoad = false;
    await click("#loadRemaining");
    await until('document.querySelectorAll(".answer-options").length === 12');
    saveMode = "offline";
    await click('.results-list input[value="yes"]');
    await click("#sendButton");
    await until(
        'document.querySelector("#status").textContent.startsWith("Save failed:")',
    );
    assert.equal(
        await evaluate(
            'document.querySelector(".results-list input[value=yes]").checked',
        ),
        true,
    );
    assert.equal(
        await evaluate('document.querySelector("#loadRemaining").disabled'),
        true,
    );
    saveMode = "badAck";
    await click("#sendButton");
    await until(
        'document.querySelector("#status").textContent.includes("acknowledgement")',
    );
    saveMode = "normal";
    await click("#sendButton");
    await until(
        'document.querySelector("#status").textContent.startsWith("Annotations saved.")',
    );
    assert.deepEqual(sent[1], sent[2]);
    assert.deepEqual(sent[2], sent[3]);
    assert.ok(!savedIds.includes(sent[3].annotations[0].windowId));
    await evaluate(
        'const select=document.querySelector("#scientific"); select.value=[...select.options].find(o=>o.value!=="DENEBR").value; select.dispatchEvent(new Event("change"));',
    );
    await until(
        'document.querySelector(".results-list").textContent.includes("No clips are waiting")',
    );
    assert.deepEqual(errors, []);
    const screenshot = await call("Page.captureScreenshot");
    await writeFile(
        ".annotation-test-local/browser-empty-state.png",
        Buffer.from(screenshot.data, "base64"),
    );
    console.log(
        "PASS: real browser references, names, null media, playback start/end/replay, audio exclusion, three labels, save, remaining queue, refresh failure, offline save, invalid ack, exact snapshot retries, empty species, no uncaught exceptions.",
    );
} finally {
    ws.close();
}
