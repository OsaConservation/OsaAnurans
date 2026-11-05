import test from "node:test";
import assert from "node:assert/strict";
import {
    DemoProvider,
    ServerProvider as ProtectedServerProvider,
    buildSubmission,
} from "../docs/annotation-data.mjs";

// Adapter tests inject transport; authentication is tested separately.
class ServerProvider extends ProtectedServerProvider {
    constructor(baseUrl, storage = null) {
        super(baseUrl, storage, {
            request: (path, options) => fetch(`${baseUrl.replace(/\/$/, "")}${path}`, options),
        });
    }
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
        const data = url.endsWith("/species?has_audio=true") ? { species: [
            { species_id: "A & B", scientific_name: "Frog A", common_name: "A" },
            { species_id: "C", scientific_name: "Frog C", common_name: "C" },
        ] } : {
            references: { "A & B": [{ reference_id: "r1", audio_url: "https://example.com/a?signature=exact", spectrogram_url: null }], C: [] },
            urls_expire_at: new Date(now + 900000).toISOString(),
        };
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
        await Promise.all([provider.listSpecies(), provider.listSpecies()]);
        assert.equal(calls.length, 1);
        const [a, c] = await Promise.all([provider.getReferenceSounds("A & B"), provider.getReferenceSounds("C")]);
        assert.equal(calls.length, 2);
        assert.deepEqual(new URL(calls[1]).searchParams.getAll("species_id"), ["A & B", "C"]);
        assert.equal(a.target.clips[0].id, "r1");
        assert.equal(a.target.clips[0].spectrogramUrl, null);
        assert.equal(a.target.clips[0].audioUrl, "https://example.com/a?signature=exact");
        assert.deepEqual(c.target.clips, []);
        assert.equal(a.comparisons[0].id, "C");
        await provider.getReferenceSounds("C");
        assert.equal(calls.length, 2);
        const restored = new ServerProvider("https://example.com/", storage);
        assert.deepEqual(await restored.getReferenceSounds("C"), c);
        assert.equal(calls.length, 2);

        now += 850000;
        const expired = new ServerProvider("https://example.com/", storage);
        await expired.getReferenceSounds("C");
        assert.equal(calls.length, 3);
        await assert.rejects(provider.getReferenceSounds("unknown"), /Unknown/);
    } finally {
        globalThis.fetch = originalFetch;
        Date.now = originalNow;
    }
});


test("candidate adapter preserves IDs, bounds, pagination, and request state", async () => {
    const originalFetch = globalThis.fetch;
    const calls = [];
    let invalid = false;
    globalThis.fetch = async (url) => {
        calls.push(new URL(url));
        return { ok: true, json: async () => ({
            candidates: [{
                candidate_id: "candidate-1", window_id: "window-1",
                query_species_id: "DENEBR", recording_id: "recording-1",
                created_at: "2026-11-05T00:00:00Z", score: 0.1,
                start_seconds: 10, end_seconds: invalid ? 9 : 15,
                audio_url: "https://example.com/audio?signature=unchanged",
                spectrogram_url: null,
            }],
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
        assert.deepEqual(batch.clips[0].offsets, [10, 15]);
        assert.equal(batch.nextCursor, null);
        assert.equal(batch.inputCursor, "opaque +/=");
        await provider.getCandidates("DENEBR");
        assert.equal(calls[1].searchParams.has("cursor"), false);
        invalid = true;
        await assert.rejects(provider.getCandidates("DENEBR"), /Invalid candidate/);
    } finally { globalThis.fetch = originalFetch; }
});
