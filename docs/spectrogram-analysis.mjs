// Frequency analysis stays independent of the DOM for testing.
export function frequencyMagnitudes(samples, offset, size = 1024) {
    if (!Number.isInteger(size) || size < 2 || (size & (size - 1)) !== 0) {
        throw new Error("FFT size must be a power of two.");
    }

    const real = new Float64Array(size);
    const imag = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        const window = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (size - 1));
        real[i] = (samples[offset + i] || 0) * window;
    }

    // Reorder samples for the iterative radix-2 transform.
    for (let i = 1, j = 0; i < size; i++) {
        let bit = size >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) [real[i], real[j]] = [real[j], real[i]];
    }

    for (let length = 2; length <= size; length *= 2) {
        const angle = -2 * Math.PI / length;
        for (let start = 0; start < size; start += length) {
            for (let j = 0; j < length / 2; j++) {
                const cosine = Math.cos(angle * j);
                const sine = Math.sin(angle * j);
                const a = start + j;
                const b = a + length / 2;
                const r = real[b] * cosine - imag[b] * sine;
                const im = real[b] * sine + imag[b] * cosine;
                real[b] = real[a] - r;
                imag[b] = imag[a] - im;
                real[a] += r;
                imag[a] += im;
            }
        }
    }

    return Float32Array.from(real.subarray(0, size / 2), (value, i) =>
        Math.hypot(value, imag[i]) * 2 / size,
    );
}

export function extractWindow(buffer, [start, end]) {
    const first = Math.round(start * buffer.sampleRate);
    const last = Math.round(end * buffer.sampleRate);
    if (!Number.isFinite(start) || !Number.isFinite(end) || first < 0 || last <= first || last > buffer.length) {
        throw new Error("Candidate window extends outside the recording.");
    }
    const samples = new Float32Array(last - first);
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
        const source = buffer.getChannelData(channel);
        for (let i = 0; i < samples.length; i++) {
            samples[i] += source[first + i] / buffer.numberOfChannels;
        }
    }
    return samples;
}

// Use a floor after normalization so silence remains at the darkest color.
export function relativeIntensity(magnitude, peak, rangeDb = 80) {
    if (!(magnitude > 0) || !(peak > 0)) return 0;

    const db = 20 * Math.log10(magnitude / peak);
    return Math.max(0, Math.min(1, (db + rangeDb) / rangeDb));
}
