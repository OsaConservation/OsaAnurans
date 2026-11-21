import test from "node:test";
import assert from "node:assert/strict";
import { extractWindow, frequencyMagnitudes } from "../docs/spectrogram-analysis.mjs";

test("spectrogram crops the candidate interval and mixes channels", () => {
    const channels = [Float32Array.from([0, 0, 2, 4, 8, 8]), Float32Array.from([0, 0, 4, 6, 8, 8])];
    const buffer = { sampleRate: 2, length: 6, numberOfChannels: 2, getChannelData: (i) => channels[i] };
    assert.deepEqual([...extractWindow(buffer, [1, 2])], [3, 5]);
    assert.throws(() => extractWindow(buffer, [0, 4]), /outside/);
});

test("FFT locates a known tone and handles silence without invalid values", () => {
    const size = 1024;
    const bin = 64;
    const tone = Float32Array.from({ length: size }, (_, i) => Math.sin(2 * Math.PI * bin * i / size));
    const spectrum = frequencyMagnitudes(tone, 0, size);
    const peak = spectrum.indexOf(Math.max(...spectrum));
    assert.equal(peak, bin);
    assert.ok(spectrum[bin] > 0.4);
    assert.ok(frequencyMagnitudes(new Float32Array(size), 0).every((value) => value === 0));
});

test("relative spectrogram colors keep silence dark and normalize weak signals", async () => {
    const { relativeIntensity } = await import("../docs/spectrogram-analysis.mjs");
    assert.equal(relativeIntensity(0, 0), 0);
    assert.equal(relativeIntensity(0, 1), 0);
    assert.equal(relativeIntensity(1e-10, 1e-10), 1);
    assert.equal(relativeIntensity(1e-4, 1), 0);
    assert.equal(relativeIntensity(0.01, 1), 0.5);
});

test("invalid FFT sizes and non-finite candidate bounds are rejected", () => {
    assert.throws(() => frequencyMagnitudes(new Float32Array(10), 0, 10), /power of two/);
    const buffer = { sampleRate: 2, length: 6 };
    assert.throws(() => extractWindow(buffer, [NaN, 2]), /outside/);
});
