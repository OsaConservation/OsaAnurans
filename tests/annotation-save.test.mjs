import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { buildSubmission } from "../docs/annotation-data.mjs";

// Exercise the actual page handlers without adding a DOM runtime dependency.
async function page() {
    const nodes = new Map();
    const node = () => ({
        textContent: "",
        options: [{}],
        disabled: false,
        listeners: {},
        addEventListener(type, handler) {
            this.listeners[type] = handler;
        },
        getBoundingClientRect: () => ({ height: 50 }),
    });
    const document = {
        querySelector(selector) {
            if (!nodes.has(selector)) nodes.set(selector, node());
            return nodes.get(selector);
        },
        querySelectorAll: () => [],
        addEventListener() {},
        documentElement: { style: { setProperty() {} } },
    };
    const context = vm.createContext({
        document,
        buildSubmission,
        crypto: { randomUUID: () => globalThis.crypto.randomUUID() },
        window: {
            annotationConfig: { provider: "server" },
            matchMedia: () => ({ addEventListener() {} }),
            addEventListener() {},
        },
        ResizeObserver: class {
            observe() {}
        },
    });
    const source = (
        await readFile(new URL("../docs/annotate.js", import.meta.url), "utf8")
    )
        .replace(/^import .*;\r?\n/gm, "")
        .replace(/initialize\(\);\s*$/, "");
    vm.runInContext(source, context);
    vm.runInContext(
        `speciesId='DENEBR'; batch={batchId:'context',clips:[{id:'w',recordingId:588,offsets:[50,55],uri:'file.wav',datasetName:'grillos'}]}; answers.set('w','yes');`,
        context,
    );
    return { nodes, context, run: (code) => vm.runInContext(code, context) };
}

test("page preserves answers and exact submission snapshot after failure and invalid ack", async () => {
    const { nodes, context, run } = await page();
    const payloads = [];
    context.mockProvider = {
        saveAnnotations: async (payload) => {
            payloads.push(JSON.parse(JSON.stringify(payload)));
            if (payloads.length === 1) throw new Error("timeout");
            if (payloads.length === 2)
                return { submissionId: "wrong", savedCount: 1 };
            return { submissionId: payload.submissionId, savedCount: 1 };
        },
    };
    run("provider=mockProvider");
    const save = nodes.get("#sendButton").listeners.click;
    for (let i = 0; i < 2; i++) {
        await save();
        assert.equal(run("answers.get('w')"), "yes");
        assert.equal(run("dirty()"), true);
        assert.equal(nodes.get("#loadRemaining").disabled, true);
        assert.match(nodes.get("#status").textContent, /Save failed/);
        await nodes.get("#loadRemaining").listeners.click();
        assert.equal(run("batch.clips.length"), 1);
    }
    await save();
    assert.deepEqual(payloads[0], payloads[1]);
    assert.deepEqual(payloads[1], payloads[2]);
    assert.equal(run("dirty()"), false);
    assert.equal(nodes.get("#loadRemaining").disabled, false);
});

test("saving locks the page and prevents duplicate saves until acknowledgement", async () => {
    const { nodes, context, run } = await page();
    let resolve;
    let calls = 0;
    context.mockProvider = {
        saveAnnotations: (payload) => {
            calls++;
            return new Promise((done) => {
                resolve = () =>
                    done({ submissionId: payload.submissionId, savedCount: 1 });
            });
        },
    };
    run("provider=mockProvider");
    const save = nodes.get("#sendButton").listeners.click;
    const saving = save();
    assert.equal(nodes.get("#sendButton").disabled, true);
    assert.equal(nodes.get("#scientific").disabled, true);
    assert.equal(nodes.get("#loadRemaining").disabled, true);
    await save();
    assert.equal(calls, 1);
    resolve();
    await saving;
    assert.equal(run("dirty()"), false);
});
