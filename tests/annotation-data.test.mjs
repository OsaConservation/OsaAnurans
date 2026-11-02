import test from "node:test";
import assert from "node:assert/strict";
import {
    DemoProvider,
    ServerProvider,
    buildSubmission,
} from "../docs/annotation-data.mjs";

test("batch includes answered windows only and preserves target and source metadata", async () => {
    const provider = new DemoProvider({});
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
test("demo retries do not duplicate a saved submission; storage errors reject", async () => {
    let stored = null;
    const provider = new DemoProvider({
        getItem: () => stored,
        setItem: (_key, value) => {
            stored = value;
        },
    });
    const batch = {
        submissionId: "retry-1",
        annotations: [{ label: "unsure" }],
    };
    assert.equal((await provider.saveAnnotations(batch)).savedCount, 1);
    await provider.saveAnnotations(batch);
    assert.equal(JSON.parse(stored).length, 1);
    const broken = new DemoProvider({
        getItem: () => null,
        setItem: () => {
            throw new Error("Storage full");
        },
    });
    await assert.rejects(broken.saveAnnotations(batch), /Storage full/);
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
        const provider = new DemoProvider({});
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
