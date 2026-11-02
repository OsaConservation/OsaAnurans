import test from "node:test";
import assert from "node:assert/strict";
import { ServerProvider, buildSubmission } from "../docs/annotation-data.mjs";

const species = {
    id: "DENEBR",
    scientificName: "Dendropsophus ebraccatus",
    commonName: "Hourglass treefrog",
};
const clip = {
    id: "opaque:window:1",
    recordingId: 588,
    offsets: [50, 55],
    uri: "sw2/file.wav",
    datasetName: "grillos",
    audioUrl: "http://localhost/media/1",
    spectrogramUrl: null,
    startSeconds: 50,
    endSeconds: 55,
    numAnnotations: 0,
};
const batch = { batchId: "context", clips: [clip] };
const refs = {
    target: {
        ...species,
        imageUrl: null,
        clips: [
            {
                audioUrl: "http://localhost/reference.mp3",
                spectrogramUrl: null,
                startSeconds: null,
                endSeconds: null,
            },
        ],
    },
    comparisons: [
        {
            id: "OTHER",
            scientificName: "Other frog",
            commonName: "",
            clips: [],
        },
    ],
};
const response = (data, status = 200) =>
    new Response(JSON.stringify(data), { status });
const providerFor = (data) =>
    new ServerProvider("http://localhost/api///", {
        fetch: async () => response(data),
    });

test("server routes, encoded opaque IDs, trailing slashes and POST JSON", async () => {
    const calls = [];
    const payload = buildSubmission(
        species.id,
        batch.batchId,
        batch.clips,
        new Map([[clip.id, "unsure"]]),
        "uuid",
    );
    const results = [
        [species],
        { ...refs, target: { ...refs.target, id: "frog /?#" } },
        batch,
        { submissionId: "uuid", savedCount: 1 },
    ];
    const provider = new ServerProvider("http://localhost/api///", {
        fetch: async (url, options) => {
            calls.push([url, options]);
            return response(results.shift());
        },
    });
    assert.equal(provider.baseUrl, "http://localhost/api");
    assert.deepEqual(await provider.listSpecies(), [species]);
    await provider.getReferenceSounds("frog /?#");
    assert.deepEqual(await provider.getCandidates("frog /?#"), batch);
    assert.deepEqual(await provider.saveAnnotations(payload), {
        submissionId: "uuid",
        savedCount: 1,
    });
    assert.deepEqual(
        calls.map(([url]) => url),
        [
            "http://localhost/api/species",
            "http://localhost/api/species/frog%20%2F%3F%23/references",
            "http://localhost/api/species/frog%20%2F%3F%23/candidates",
            "http://localhost/api/annotations",
        ],
    );
    assert.equal(calls[3][1].method, "POST");
    assert.equal(calls[3][1].headers["Content-Type"], "application/json");
    assert.deepEqual(JSON.parse(calls[3][1].body), payload);
    assert.equal(calls[0][1].signal.aborted, false);
});

test("null media, empty reference lists and empty queue are valid", async () => {
    assert.deepEqual(
        await providerFor(refs).getReferenceSounds("DENEBR"),
        refs,
    );
    assert.deepEqual(
        await providerFor({ batchId: null, clips: [] }).getCandidates("DENEBR"),
        { batchId: null, clips: [] },
    );
});

test("invalid species, references and candidate invariants reject", async () => {
    for (const data of [
        { species: [] },
        [species, species],
        [{ ...species, id: 1 }],
        [{ ...species, scientificName: null }],
    ])
        await assert.rejects(
            providerFor(data).listSpecies(),
            /Invalid server response/,
        );
    for (const data of [
        null,
        { ...refs, target: { ...refs.target, id: "wrong" } },
        { ...refs, comparisons: [refs.target] },
        { ...refs, target: { ...refs.target, clips: [{}] } },
        {
            ...refs,
            target: {
                ...refs.target,
                clips: [{ audioUrl: "x", startSeconds: 2, endSeconds: null }],
            },
        },
    ])
        await assert.rejects(
            providerFor(data).getReferenceSounds("DENEBR"),
            /Invalid server response/,
        );
    const badClips = [
        { ...clip, id: 4 },
        { ...clip, recordingId: "588" },
        { ...clip, uri: null },
        { ...clip, offsets: [1, 2] },
        { ...clip, endSeconds: 50 },
        { ...clip, numAnnotations: -1 },
        { ...clip, numAnnotations: 0.5 },
        { ...clip, spectrogramUrl: 1 },
    ];
    for (const data of [
        null,
        { batchId: null, clips: [clip] },
        { ...batch, clips: [clip, clip] },
        { ...batch, clips: Array(16).fill(clip) },
        ...badClips.map((c) => ({ ...batch, clips: [c] })),
    ])
        await assert.rejects(
            providerFor(data).getCandidates("DENEBR"),
            /Invalid server response/,
        );
});

test("HTTP statuses, non-JSON and network failures remain errors", async () => {
    for (const status of [404, 409, 422, 503]) {
        const provider = new ServerProvider("http://localhost", {
            fetch: async () =>
                response({ error: { message: "Useful detail" } }, status),
        });
        await assert.rejects(
            provider.listSpecies(),
            new RegExp(`${status}: Useful detail`),
        );
    }
    for (const status of [200, 503]) {
        const provider = new ServerProvider("http://localhost", {
            fetch: async () => new Response("<html>", { status }),
        });
        await assert.rejects(provider.listSpecies(), /JSON/);
    }
    await assert.rejects(
        new ServerProvider("x", {
            fetch: async () => {
                throw new Error("offline");
            },
        }).listSpecies(),
        /offline/,
    );
});

test("finite timeout aborts fetch and covers a stalled response body", async () => {
    for (const stalledBody of [false, true]) {
        let signal;
        const provider = new ServerProvider("x", {
            timeoutMs: 10,
            fetch: async (_url, options) => {
                signal = options.signal;
                if (stalledBody)
                    return { ok: true, json: () => new Promise(() => {}) };
                return new Promise(() => {});
            },
        });
        await assert.rejects(provider.listSpecies(), /timed out/);
        assert.equal(signal.aborted, true);
    }
});

test("save rejects missing, mismatched and partial acknowledgements", async () => {
    const payload = { submissionId: "uuid", annotations: [{}] };
    for (const data of [
        null,
        {},
        { submissionId: "other", savedCount: 1 },
        { submissionId: "uuid", savedCount: 0 },
        { submissionId: "uuid", savedCount: "1" },
    ])
        await assert.rejects(
            providerFor(data).saveAnnotations(payload),
            /acknowledgement/,
        );
});
