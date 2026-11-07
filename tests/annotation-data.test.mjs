import test from "node:test";
import assert from "node:assert/strict";
import {
    DemoProvider,
    createProvider,
    CandidateCursorError,
    ServerProvider as ProtectedServerProvider,
    buildSubmission,
} from "../docs/annotation-data.mjs";
import { candidatePlayer } from "../docs/candidate-player.mjs";
import { extractWindow } from "../docs/spectrogram-analysis.mjs";

// Adapter tests inject transport; authentication is tested separately.
class ServerProvider extends ProtectedServerProvider {
    constructor(baseUrl, storage = null) {
        super(baseUrl, storage, {
            request: (path, options) => fetch(`${baseUrl.replace(/\/$/, "")}${path}`, options),
        });
    }
}

function serverCandidate(overrides = {}) {
    return {
        candidate_id: "candidate-1", window_id: "window-1",
        query_species_id: "DENEBR", recording_id: "recording-1",
        created_at: "2026-11-05T00:00:00Z", score: 0.1,
        start_seconds: 0, end_seconds: 5,
        recording_start_seconds: 120, recording_end_seconds: 125,
        audio_url: "https://example.com/windows/window-1.wav?signature=unchanged",
        spectrogram_url: null,
        ...overrides,
    };
}

test("annotation page providers do not persist catalog caches", () => {
    const provider = createProvider({ provider: "server", baseUrl: "https://example.com" });
    assert.equal(provider.storage, null);
    assert.equal(provider.species, undefined);
    assert.equal(provider.references, undefined);
});

test("outdated cursors can restart a batch without retrying the same invalid token", async () => {
    const api = { request: async () => ({ ok: false, status: 400 }) };
    const provider = new ProtectedServerProvider("https://example.com", null, api);
    await assert.rejects(provider.getCandidates("DENEBR", { cursor: "old-v1-cursor" }),
        (error) => error instanceof CandidateCursorError && /first batch/.test(error.message));
    await assert.rejects(provider.getCandidates("DENEBR"),
        (error) => !(error instanceof CandidateCursorError) && error.status === 400);
});

test("empty restored cursors restart once without changing normal pagination", async () => {
    const paths = [];
    const api = {
        request: async (path) => {
            paths.push(new URL(path, "https://example.com"));
            return { ok: true, json: async () => ({
                candidates: path.includes("cursor=") ? [] : [serverCandidate()],
                next_cursor: null, urls_expire_at: "2099-01-01T00:00:00Z",
            }) };
        },
    };
    const provider = new ProtectedServerProvider("https://example.com", null, api);
    const batch = await provider.getCandidates("DENEBR", {
        count: 5, cursor: "old-queue", restartIfEmpty: true,
    });
    assert.equal(batch.clips.length, 1);
    assert.equal(batch.inputCursor, null);
    assert.equal(paths.length, 2);
    assert.equal(paths[1].searchParams.has("cursor"), false);
    assert.equal(paths[1].searchParams.get("count"), "5");

    const end = await provider.getCandidates("DENEBR", { cursor: "end" });
    assert.equal(end.clips.length, 0);
    assert.equal(paths.length, 3);
});

async function loadServerClip(overrides = {}) {
    const api = {
        request: async () => ({
            ok: true,
            json: async () => ({
                candidates: [serverCandidate(overrides)],
                next_cursor: null,
                urls_expire_at: "2099-01-01T00:00:00Z",
            }),
        }),
    };
    const provider = new ProtectedServerProvider("https://example.com", null, api);
    const batch = await provider.getCandidates("DENEBR");
    return batch.clips[0];
}

test("batch includes answered windows only and preserves target and source metadata", async () => {
    const provider = new DemoProvider();
    const { batchId, clips } = await provider.getCandidates("frog-a");
    const payload = buildSubmission(
        "frog-a",
        batchId,
        clips,
        new Map([
            [clips[0].id, "yes"],
            [clips[2].id, "no"],
            [clips[4].id, "unsure"],
        ]),
        "submission-1",
    );
    assert.equal(payload.speciesId, "frog-a");
    assert.equal(payload.batchId, "demo:frog-a");
    assert.deepEqual(
        payload.annotations.map((row) => row.label),
        ["yes", "no", "unsure"],
    );
    assert.deepEqual(payload.annotations[0].offsets, [35, 40]);
    assert.equal(payload.annotations[0].recordingId, 633);
    assert.equal(payload.annotations[0].windowId, "25435");
});
test("demo saving is explicitly unavailable", async () => {
    const provider = new DemoProvider();
    await assert.rejects(provider.saveAnnotations({ annotations: [] }), /unavailable in demo mode/);
});

test("server submissions validate the complete receipt", async () => {
    const original = globalThis.fetch;
    const submission = { submission_id: "test", annotations: [{ response: "present" }] };
    let accepted = 1;
    globalThis.fetch = async (_url, options) => {
        assert.equal(options.method, "POST");
        assert.deepEqual(JSON.parse(options.body), submission);
        return { ok: true, json: async () => ({ submission_id: "test", accepted_count: accepted }) };
    };
    try {
        const provider = new ServerProvider("https://example.com");
        await provider.saveAnnotations(submission);
        accepted = 0;
        await assert.rejects(provider.saveAnnotations(submission), /uncertain/);
    } finally { globalThis.fetch = original; }
});

test("reference and candidate audio share clips and media fields", async () => {
    const originalParser = globalThis.Papa;
    globalThis.Papa = {
        parse(source, options) {
            if (options?.download) {
                options.complete({
                    errors: [],
                    data: [{
                        display: "1",
                        scientific_name: "Test frog",
                        common_name: "Test frog",
                        audio_files: "reference.wav",
                        spec_files: "reference.png",
                    }],
                });
                return;
            }
            return { data: [[source]] };
        },
    };
    try {
        const provider = new DemoProvider();
        const references = await provider.getReferenceSounds("Test frog");
        const candidates = await provider.getCandidates("Test frog");
        assert.ok(Array.isArray(references.target.clips));
        assert.ok(Array.isArray(candidates.clips));
        assert.equal(references.target.clips.length, 1);
        assert.equal(candidates.clips.length, 5);
        assert.equal("examples" in references.target, false);
        assert.equal("recordings" in candidates, false);
        for (const clip of [...references.target.clips, ...candidates.clips]) {
            assert.equal(typeof clip.audioUrl, "string");
            assert.equal(typeof clip.spectrogramUrl, "string");
        }
        assert.equal(candidates.clips[0].recordingId, 633);
    } finally {
        if (originalParser === undefined) delete globalThis.Papa;
        else globalThis.Papa = originalParser;
    }
});

test("server caches catalog and references, refreshes expired URLs, and retries failures", async () => {
    const originalFetch = globalThis.fetch;
    const originalNow = Date.now;
    let now = originalNow();
    let calls = [];
    let fail = false;
    Date.now = () => now;
    globalThis.fetch = async (url, options) => {
        calls.push(url);
        assert.equal(options.credentials, "omit");
        if (fail) return { ok: false, status: 500 };
        const data = url.includes("/species?") ? { species: [
            { species_id: "A & B", scientific_name: "Frog A", common_name: "A" },
            { species_id: "C", scientific_name: "Frog C", common_name: "C" },
        ] } : {
            references: { "A & B": [{ reference_id: "r1", audio_url: "https://example.com/a?signature=exact", spectrogram_url: null }], C: [] },
            urls_expire_at: new Date(now + 900000).toISOString(),
        };
        // C has reference media but no candidates, so it remains a comparison only.
        if (url.includes("has_candidates=true")) data.species = data.species.slice(0, 1);
        return { ok: true, json: async () => data };
    };
    try {
        let stored = null;
        const storage = {
            getItem: () => stored,
            setItem: (_key, value) => { stored = value; },
        };
        const provider = new ServerProvider("https://example.com/", storage);
        fail = true;
        await assert.rejects(provider.listSpecies(), /500/);
        fail = false;
        calls = [];
        const [targets] = await Promise.all([provider.listSpecies(), provider.listSpecies()]);
        assert.deepEqual(targets.map((species) => species.id), ["A & B"]);
        assert.equal(calls.length, 1);
        assert.equal(new URL(calls[0]).searchParams.get("has_candidates"), "true");
        const [a, c] = await Promise.all([provider.getReferenceSounds("A & B"), provider.getReferenceSounds("C")]);
        assert.equal(calls.length, 3);
        assert.equal(new URL(calls[1]).searchParams.has("has_candidates"), false);
        assert.deepEqual(new URL(calls[2]).searchParams.getAll("species_id"), ["A & B", "C"]);
        assert.equal(a.target.clips[0].id, "r1");
        assert.equal(a.target.clips[0].spectrogramUrl, null);
        assert.equal(a.target.clips[0].audioUrl, "https://example.com/a?signature=exact");
        assert.deepEqual(c.target.clips, []);
        assert.equal(a.comparisons[0].id, "C");
        await provider.getReferenceSounds("C");
        assert.equal(calls.length, 3);
        const restored = new ServerProvider("https://example.com/", storage);
        assert.deepEqual(await restored.getReferenceSounds("C"), c);
        assert.equal(calls.length, 3);

        now += 850000;
        const expired = new ServerProvider("https://example.com/", storage);
        await expired.getReferenceSounds("C");
        assert.equal(calls.length, 4);
        await assert.rejects(provider.getReferenceSounds("unknown"), /Unknown/);
    } finally {
        globalThis.fetch = originalFetch;
        Date.now = originalNow;
    }
});


test("candidate adapter separates playback and recording bounds and preserves opaque pagination", async () => {
    const originalFetch = globalThis.fetch;
    const calls = [];
    let invalid = false;
    globalThis.fetch = async (url) => {
        calls.push(new URL(url));
        return { ok: true, json: async () => ({
            candidates: [serverCandidate({ end_seconds: invalid ? 9 : 5 })],
            next_cursor: null, urls_expire_at: "2099-01-01T00:00:00Z",
        }) };
    };
    try {
        const provider = new ServerProvider("https://example.com", null);
        const batch = await provider.getCandidates("DENEBR", { count: 2, cursor: "opaque +/=" });
        assert.equal(calls[0].searchParams.get("cursor"), "opaque +/=");
        assert.equal(calls[0].searchParams.get("count"), "2");
        assert.equal(batch.clips[0].id, "candidate-1");
        assert.equal(batch.clips[0].windowId, "window-1");
        assert.deepEqual(batch.clips[0].offsets, [0, 5]);
        assert.deepEqual(batch.clips[0].recordingOffsets, [120, 125]);
        assert.equal(batch.clips[0].audioUrl, "https://example.com/windows/window-1.wav?signature=unchanged");
        assert.equal(batch.nextCursor, null);
        assert.equal(batch.inputCursor, "opaque +/=");
        await provider.getCandidates("DENEBR");
        assert.equal(calls[1].searchParams.has("cursor"), false);
        invalid = true;
        await assert.rejects(provider.getCandidates("DENEBR"), /Invalid candidate/);
    } finally { globalThis.fetch = originalFetch; }
});

test("candidate adapter rejects legacy recording playback bounds and missing source metadata", async () => {
    await assert.rejects(loadServerClip({ start_seconds: 120, end_seconds: 125 }), /standalone 5-second/);
    await assert.rejects(loadServerClip({ start_seconds: 0, end_seconds: 6 }), /standalone 5-second/);
    for (const overrides of [
        { recording_start_seconds: undefined },
        { recording_end_seconds: undefined },
        { recording_start_seconds: null },
        { recording_start_seconds: -1, recording_end_seconds: 4 },
        { recording_start_seconds: 120, recording_end_seconds: 119 },
        { recording_end_seconds: Infinity },
        { recording_end_seconds: 126 },
    ]) {
        await assert.rejects(loadServerClip(overrides), /source-recording bounds/);
    }
    const clip = await loadServerClip({ recording_start_seconds: 0.1, recording_end_seconds: 5.1 });
    assert.deepEqual(clip.offsets, [0, 5]);
    assert.deepEqual(clip.recordingOffsets, [0.1, 5.1]);
});

test("standalone candidates play and crop spectrogram audio from zero despite nonzero recording offsets", async () => {
    const clip = await loadServerClip({ spectrogram_url: "https://example.com/window.png" });
    const originalDocument = globalThis.document;

    class Element extends EventTarget {
        constructor(tagName) {
            super();
            this.tagName = tagName;
            this.children = [];
            this.currentTime = 120;
            this.readyState = 1;
            this.paused = true;
        }
        append(...nodes) { this.children.push(...nodes); }
        setAttribute() {}
    }

    globalThis.document = { createElement: (tagName) => new Element(tagName) };
    try {
        const player = candidatePlayer(clip, { ensureFresh: async () => true, onFailure: () => {} });
        const audio = player.children.find((element) => element.tagName === "audio");
        const seek = player.children.find((element) => element.tagName === "input");
        audio.dispatchEvent(new Event("loadedmetadata"));
        assert.equal(audio.currentTime, 0);
        assert.equal(seek.max, 5);
        seek.value = "3.25";
        seek.dispatchEvent(new Event("input"));
        assert.equal(audio.currentTime, 3.25);

        const samples = Float32Array.from({ length: 10 }, (_, index) => index + 1);
        const buffer = { sampleRate: 2, length: 10, numberOfChannels: 1, getChannelData: () => samples };
        assert.deepEqual(extractWindow(buffer, clip.offsets), samples);
        assert.throws(() => extractWindow(buffer, clip.recordingOffsets), /outside/);
    } finally {
        if (originalDocument === undefined) delete globalThis.document;
        else globalThis.document = originalDocument;
    }
});
