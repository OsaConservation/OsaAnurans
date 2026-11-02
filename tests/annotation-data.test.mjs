import test from "node:test";
import assert from "node:assert/strict";
import {
    DemoProvider,
    ServerProvider,
    buildSubmission,
} from "../docs/annotation-data.mjs";

test("batch includes answered windows only and preserves target and source metadata", async () => {
    const provider = new DemoProvider({});
    const { batchId, recordings } = await provider.getRecordings("frog-a");
    const payload = buildSubmission(
        "frog-a",
        batchId,
        recordings,
        new Map([
            [recordings[0].id, "yes"],
            [recordings[2].id, "no"],
            [recordings[4].id, "unsure"],
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
test("server skeleton never reports a successful operation", async () => {
    const provider = new ServerProvider("http://localhost:8080/api/");
    assert.equal(provider.baseUrl, "http://localhost:8080/api");
    for (const operation of [
        "listSpecies",
        "getReferenceSounds",
        "getRecordings",
        "saveAnnotations",
    ]) {
        await assert.rejects(provider[operation]("frog"), /not implemented/);
    }
});
