import test from "node:test";
import assert from "node:assert/strict";
import { limitPlayback } from "../docs/audio-window.mjs";

class Audio extends EventTarget {
    currentTime = 0;
    paused = true;
    seeking = false;
    readyState = 4;
    playbackRate = 1;
    pause() { this.paused = true; this.emit("pause"); }
    play() { this.paused = false; this.emit("play"); this.emit("playing"); }
    emit(name) { this.dispatchEvent(new Event(name)); }
}
function fixture() {
    const audio = new Audio();
    const timers = new Map();
    let nextId = 0;
    const clock = {
        setTimeout(fn, delay) { timers.set(++nextId, { fn, delay }); return nextId; },
        clearTimeout(id) { timers.delete(id); },
    };
    limitPlayback(audio, 10, 15, clock);
    return { audio, timers };
}

test("starts at the window, stops at its end, and replays from the start", () => {
    const { audio, timers } = fixture();
    audio.emit("loadedmetadata");
    assert.equal(audio.currentTime, 10);
    audio.play();
    const timer = [...timers.values()][0];
    assert.equal(timer.delay, 5000);
    timer.fn();
    assert.equal(audio.paused, true);
    assert.equal(audio.currentTime, 15);
    audio.play();
    assert.equal(audio.currentTime, 10);
    audio.pause();
    assert.equal(timers.size, 0);
});

test("clamps seeking and stops on time updates beyond the window", () => {
    const { audio, timers } = fixture();
    audio.play();
    audio.currentTime = 2;
    audio.emit("seeking");
    assert.equal(audio.currentTime, 10);
    assert.equal(timers.size, 0);
    audio.emit("seeked");
    assert.equal(timers.size, 1);
    audio.currentTime = 16;
    audio.emit("timeupdate");
    assert.equal(audio.paused, true);
    assert.equal(audio.currentTime, 15);
});

test("buffering suspends the stop timer; resume and playback speed reschedule it", () => {
    const { audio, timers } = fixture();
    audio.play();
    audio.currentTime = 12;
    audio.emit("waiting");
    assert.equal(timers.size, 0);
    audio.emit("playing");
    assert.equal([...timers.values()][0].delay, 3000);
    audio.playbackRate = 2;
    audio.emit("ratechange");
    assert.equal([...timers.values()][0].delay, 1500);
    audio.emit("error");
    assert.equal(timers.size, 0);
});

test("invalid window bounds fail immediately", () => {
    for (const offsets of [[-1, 3], [2, 2], [4, 3], [0, NaN]])
        assert.throws(() => limitPlayback(new Audio(), ...offsets), /Invalid audio window/);
});
