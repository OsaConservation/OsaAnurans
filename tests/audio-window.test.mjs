import test from "node:test";
import assert from "node:assert/strict";
import { waitForMetadata, windowPosition } from "../docs/audio-window.mjs";

class FakeAudio extends EventTarget {
    readyState = 0;
    loads = 0;
    load() { this.loads++; }
}

test("metadata loading handles ready media, failure, and bounded waiting", async () => {
    const audio = new FakeAudio();
    const pending = waitForMetadata(audio);
    assert.equal(audio.loads, 1);
    audio.dispatchEvent(new Event("loadedmetadata"));
    await pending;

    const failure = waitForMetadata(audio);
    audio.dispatchEvent(new Event("error"));
    await assert.rejects(failure, /could not be loaded/);
    await assert.rejects(waitForMetadata(audio, 5), /timed out/);

    audio.readyState = 1;
    const previousLoads = audio.loads;
    await waitForMetadata(audio);
    assert.equal(audio.loads, previousLoads);
});

test("candidate timeline uses window-relative seconds and clamps both ends", () => {
    assert.equal(windowPosition(5, [10, 15]), 0);
    assert.equal(windowPosition(12, [10, 15]), 2);
    assert.equal(windowPosition(20, [10, 15]), 5);
});
