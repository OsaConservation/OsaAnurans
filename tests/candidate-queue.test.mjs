import test from "node:test";
import assert from "node:assert/strict";
import { CandidateQueue } from "../docs/candidate-queue.mjs";
import { ServerProvider } from "../docs/annotation-data.mjs";

function provider(total = 100) {
    let calls = 0;
    return {
        get calls() { return calls; },
        async getCandidates(species, { count, cursor }) {
            calls++;
            assert.equal(count, 20);
            const start = Number(cursor || 0);
            const end = Math.min(start + count, total);
            return {
                clips: Array.from({ length: end - start }, (_, i) => ({ id: String(start + i) })),
                nextCursor: end < total ? String(end) : null,
                urlsExpireAt: "2099-01-01T00:00:00Z",
            };
        },
    };
}

for (const size of [1, 5, 7, 20]) {
    test(`metadata queue consumes 100 unique clips with five queries, visible size ${size}`, async () => {
        const api = provider();
        const queue = new CandidateQueue(api, "FROG");
        const ids = [];
        let batch;
        do {
            batch = await queue.next(size);
            assert.ok(batch.clips.length <= size);
            ids.push(...batch.clips.map((clip) => clip.id));
        } while (batch.hasNext);
        assert.deepEqual(ids, Array.from({ length: 100 }, (_, i) => String(i)));
        assert.equal(api.calls, 5);
    });
}

test("failed refill and aborted reads cannot consume buffered questions", async () => {
    const api = provider(30);
    const queue = new CandidateQueue(api, "FROG");
    await queue.next(7);
    await queue.next(7);
    const original = api.getCandidates;
    api.getCandidates = async () => { throw new Error("offline"); };
    await assert.rejects(queue.next(7), /offline/);
    assert.equal(queue.buffer.length, 6);
    api.getCandidates = original;
    const abort = new AbortController();
    abort.abort();
    await assert.rejects(queue.next(7, abort.signal), { name: "AbortError" });
    assert.equal(queue.buffer.length, 6);
    assert.deepEqual((await queue.next(7)).clips.map((clip) => clip.id), ["14", "15", "16", "17", "18", "19", "20"]);
});

test("expired buffered metadata refreshes exact visible questions before advancing", async () => {
    const api = provider(25);
    const original = api.getCandidates;
    api.getCandidates = async (...args) => ({ ...await original(...args), urlsExpireAt: "2000-01-01T00:00:00Z" });
    let refreshed = [];
    api.refreshCandidateMedia = async (batch) => {
        refreshed = batch.clips.map((clip) => clip.id);
        return { ...batch, urlsExpireAt: "2099-01-01T00:00:00Z" };
    };
    const queue = new CandidateQueue(api, "FROG");
    const batch = await queue.next(5);
    assert.deepEqual(refreshed, ["0", "1", "2", "3", "4"]);
    assert.equal(api.calls, 1);
    assert.equal(batch.hasNext, true);
    assert.equal(batch.urlsExpireAt, "2099-01-01T00:00:00Z");
});

function candidate(id = "a", changes = {}) {
    return { candidate_id: id, window_id: `window-${id}`, query_species_id: "FROG",
        recording_id: "recording", recording_start_seconds: 30, recording_end_seconds: 35,
        start_seconds: 0, end_seconds: 5, audio_url: `https://media/${id}`, ...changes };
}

test("exact-ID refresh preserves questions/order and rejects missing, duplicate, or changed IDs", async () => {
    let items = [candidate("b"), candidate("a")];
    const api = new ServerProvider("https://example.com", null, {
        request: async (path) => {
            const params = new URL(path, "https://example.com").searchParams;
            assert.deepEqual(params.getAll("candidate_id"), ["a", "b"]);
            assert.equal(params.has("count"), false);
            return { ok: true, json: async () => ({ candidates: items, urls_expire_at: "2099-01-01T00:00:00Z" }) };
        },
    });
    const batch = { speciesId: "FROG", hasNext: true, clips: ["a", "b"].map((id) => ({
        id, windowId: `window-${id}`, recordingId: "recording", recordingOffsets: [30, 35], audioUrl: "old",
    })) };
    const refreshed = await api.refreshCandidateMedia(batch);
    assert.deepEqual(refreshed.clips.map((clip) => clip.id), ["a", "b"]);
    assert.equal(refreshed.hasNext, true);
    assert.equal(batch.clips[0].audioUrl, "old");
    for (const invalid of [[candidate("a")], [candidate("a"), candidate("a")],
        [candidate("a"), candidate("other")], [candidate("a", { window_id: "wrong" }), candidate("b")]]) {
        items = invalid;
        await assert.rejects(api.refreshCandidateMedia(batch));
    }
});

test("one page seed survives queue restarts, species changes, pagination, and failed retries", async () => {
    const seed = "11111111-1111-4111-8111-111111111111";
    const calls = [];
    let fail = true;
    const api = { getCandidates: async (species, options) => {
        calls.push({ species, ...options });
        if (fail) { fail = false; throw new Error("offline"); }
        return { clips: Array.from({ length: 20 }, (_, i) => ({ id: `${species}-${options.cursor || 'first'}-${i}` })),
            nextCursor: options.cursor ? null : "next", urlsExpireAt: "2099-01-01T00:00:00Z" };
    } };
    const queue = new CandidateQueue(api, "FROG", seed);
    await assert.rejects(queue.next(20), /offline/);
    await queue.next(20);
    await queue.next(20);
    await new CandidateQueue(api, "OTHER", seed).next(5);
    await new CandidateQueue(api, "FROG", seed).next(7);
    assert.ok(calls.every((call) => call.seed === seed));
    assert.deepEqual(calls.slice(0, 3).map((call) => call.cursor), [null, null, "next"]);
});

test("server provider preserves sampling seed in first requests, retries, and cursor resets", async () => {
    const paths = [];
    const seed = "11111111-1111-4111-8111-111111111111";
    const api = new ServerProvider("https://example.com", null, {
        request: async (path) => {
            paths.push(new URL(path, "https://example.com"));
            return { ok: true, json: async () => ({ candidates: [], next_cursor: null, urls_expire_at: "2099-01-01T00:00:00Z" }) };
        },
    });
    await api.getCandidates("FROG", { seed, cursor: "old", restartIfEmpty: true });
    assert.equal(paths.length, 2);
    assert.ok(paths.every((path) => path.searchParams.get("seed") === seed));
    assert.equal(paths[1].searchParams.has("cursor"), false);
});
