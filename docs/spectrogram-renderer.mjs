import { extractWindow, frequencyMagnitudes, relativeIntensity } from "./spectrogram-analysis.mjs";

// Approximate the reference images' magma palette.
const colorStops = [
    [0, 0, 4], [20, 14, 54], [59, 15, 112], [100, 26, 128],
    [140, 41, 129], [183, 55, 121], [222, 73, 104], [247, 112, 92],
    [254, 159, 109], [254, 207, 146], [252, 253, 191],
];

function spectrogramColor(value) {
    const position = Math.max(0, Math.min(1, value)) * (colorStops.length - 1);
    const index = Math.min(colorStops.length - 2, Math.floor(position));
    const fraction = position - index;
    return colorStops[index].map((channel, i) =>
        Math.round(channel + fraction * (colorStops[index + 1][i] - channel)),
    );
}

function tickStep(range, targetCount) {
    const raw = range / targetCount;
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    const normalized = raw / magnitude;
    const step = [1, 2, 2.5, 5, 10].find((value) => value >= normalized);
    return step * magnitude;
}

function drawAxes(context, plot, duration, maxFrequency) {
    const { left, top, width, height } = plot;
    context.strokeStyle = "#303038";
    context.fillStyle = "#303038";
    context.font = "14px sans-serif";
    context.lineWidth = 1;
    context.strokeRect(left, top, width, height);

    function ticks(range, majorStep, horizontal) {
        const minorStep = majorStep / 5;
        for (let i = 0; i * minorStep <= range + minorStep * 0.001; i++) {
            const value = i * minorStep;
            const major = i % 5 === 0;
            const x = horizontal ? left + value / range * width : left;
            const y = horizontal ? top + height : top + height - value / range * height;
            const length = major ? 7 : 3;
            context.beginPath();
            context.moveTo(x, y);
            context.lineTo(horizontal ? x : x - length, horizontal ? y + length : y);
            context.stroke();
            if (!major) continue;

            const label = Number(value.toFixed(3)).toString();
            context.textAlign = horizontal ? "center" : "right";
            context.textBaseline = horizontal ? "top" : "middle";
            context.fillText(label, horizontal ? x : x - 12, horizontal ? y + 12 : y);
        }
    }

    ticks(duration, tickStep(duration, 8), true);
    ticks(maxFrequency / 1000, tickStep(maxFrequency / 1000, 8), false);
    context.textAlign = "center";
    context.textBaseline = "bottom";
    context.fillText("Time within clip (s)", left + width / 2, top + height + 58);
    context.save();
    context.translate(18, top + height / 2);
    context.rotate(-Math.PI / 2);
    context.fillText("Frequency (kHz)", 0, 0);
    context.restore();
}

export async function drawSpectrogram(buffer, offsets) {
    const samples = extractWindow(buffer, offsets);
    const canvas = document.createElement("canvas");
    canvas.width = 1100;
    canvas.height = 560;
    const context = canvas.getContext("2d");
    const plot = { left: 78, top: 16, width: 1000, height: 470 };
    const fftSize = 2048;
    const maxFrequency = Math.min(12000, buffer.sampleRate / 2);
    const spectra = [];
    let peak = 0;

    for (let x = 0; x < plot.width; x++) {
        const offset = Math.floor(x * Math.max(0, samples.length - fftSize) / (plot.width - 1));
        const magnitudes = frequencyMagnitudes(samples, offset, fftSize);
        spectra.push(magnitudes);
        for (const magnitude of magnitudes) peak = Math.max(peak, magnitude);
        if (x % 32 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }

    // Match the references: 80 dB below this clip's peak maps to black.
    const pixels = context.createImageData(plot.width, plot.height);
    for (let x = 0; x < plot.width; x++) {
        for (let y = 0; y < plot.height; y++) {
            const hz = (1 - y / (plot.height - 1)) * maxFrequency;
            const bin = Math.min(spectra[x].length - 1, Math.round(hz * fftSize / buffer.sampleRate));
            const rgb = spectrogramColor(relativeIntensity(spectra[x][bin], peak));
            const index = (y * plot.width + x) * 4;
            pixels.data.set([...rgb, 255], index);
        }
        if (x % 32 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }

    context.fillStyle = "#fff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.putImageData(pixels, plot.left, plot.top);
    drawAxes(context, plot, offsets[1] - offsets[0], maxFrequency);
    return canvas.toDataURL("image/png");
}

