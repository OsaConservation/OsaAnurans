import test from "node:test";
import assert from "node:assert/strict";
import { AnnotatorSession, SessionExpiredError } from "../docs/annotator-session.mjs";
import { createSubmission, SubmissionDraft } from "../docs/annotation-submission.mjs";

test("auth gates requests, preserves password whitespace, and limits token destinations", async () => {
    const original = globalThis.fetch;
    const calls = [];
    let unauthorized = false;
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        if (url.endsWith("/auth/login")) return { ok: true, json: async () => ({ access_token: "test-token", token_type: "Bearer", expires_at: "2099-01-01T00:00:00Z" }) };
        return { ok: !unauthorized, status: unauthorized ? 401 : 200 };
    };
    try {
        const auth = new AnnotatorSession("https://example.com");
        await assert.rejects(auth.request("/species"), SessionExpiredError);
        assert.equal(calls.length, 0);
        await auth.login(" password ");
        assert.equal(JSON.parse(calls[0].options.body).password, " password ");
        for (const route of ["/species", "/references?species_id=A", "/candidates?species_id=A", "/annotations"]) {
            await auth.request(route);
            assert.equal(calls.at(-1).options.headers.get("Authorization"), "Bearer test-token");
        }
        await assert.rejects(auth.request("https://storage.googleapis.com/file"), /Unsupported/);
        unauthorized = true;
        await assert.rejects(auth.request("/species"), SessionExpiredError);
        assert.equal(auth.authenticated, false);
        await assert.rejects(auth.request("/species"), SessionExpiredError);
    } finally { globalThis.fetch = original; }
});

test("submission payload maps labels and preserves pending payload across reload", () => {
    const clips = [{ id: "candidate", windowId: "window", querySpeciesId: "FROG", audioUrl: "secret-url" }];
    const payload = createSubmission(clips, new Map([["candidate", "yes"]]), "annotator", " Maria ", "submission");
    assert.deepEqual(payload.annotations, [{ candidate_id: "candidate", window_id: "window", query_species_id: "FROG", response: "present" }]);
    assert.equal(payload.annotator_name, "Maria");
    assert.ok(!JSON.stringify(payload).includes("secret-url"));
    let stored;
    const storage = { getItem: () => stored, setItem: (_key, value) => { stored = value; } };
    new SubmissionDraft(storage, "draft").write({ pending: payload });
    assert.deepEqual(new SubmissionDraft(storage, "draft").read().pending, payload);
    assert.throws(() => createSubmission(clips, new Map(), "annotator", "Maria", "id"), /Answer between/);
});

test("login errors are distinct and never retried automatically", async () => {
    const original = globalThis.fetch;
    let status = 401;
    let calls = 0;
    globalThis.fetch = async () => { calls++; return { ok: false, status }; };
    try {
        const auth = new AnnotatorSession("https://example.com");
        await assert.rejects(auth.login("wrong"), /Incorrect password/);
        assert.equal(calls, 1);
        status = 503;
        await assert.rejects(auth.login("wrong"), /service unavailable/);
        assert.equal(calls, 2);
        assert.equal(auth.authenticated, false);
    } finally { globalThis.fetch = original; }
});
