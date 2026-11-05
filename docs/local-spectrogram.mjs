import { whenVisible } from "./observe-visible.mjs";
import { drawSpectrogram } from "./spectrogram-renderer.mjs";

const recordings = new Map();
const images = new Map();
const MAX_DECODED_BYTES = 64 * 1024 * 1024;
let decodedBytes = 0;
let queue = Promise.resolve();

function recordingKey(clip) {
    // Signed query parameters change; the object path identifies the audio.
    const url = new URL(clip.audioUrl, document.baseURI);
    return `${clip.recordingId}:${url.origin}${url.pathname}`;
}

async function decodeRecording(clip, onStatus) {
    const key = recordingKey(clip);
    if (recordings.has(key)) return recordings.get(key).buffer;

    onStatus("Downloading audio for spectrogram...");
    const response = await fetch(clip.audioUrl, {
        credentials: "omit",
        signal: AbortSignal.timeout(60_000),
    });
    if (!response.ok) throw new Error(`Audio download failed (${response.status}).`);

    const bytes = await response.arrayBuffer();
    onStatus("Decoding audio...");
    const context = new AudioContext();
    let buffer;
    try {
        buffer = await context.decodeAudioData(bytes);
    } finally {
        await context.close();
    }

    const size = buffer.length * buffer.numberOfChannels * 4;
    while (decodedBytes + size > MAX_DECODED_BYTES && recordings.size) {
        const oldest = recordings.keys().next().value;
        decodedBytes -= recordings.get(oldest).size;
        recordings.delete(oldest);
    }
    if (size <= MAX_DECODED_BYTES) {
        recordings.set(key, { buffer, size });
        decodedBytes += size;
    }
    return buffer;
}

export function generateSpectrogram(clip, onStatus = () => {}, canGenerate = async () => true) {
    const key = `${recordingKey(clip)}:${clip.offsets.join(":")}`;
    if (images.has(key)) return images.get(key);

    // Serialize generation to bound simultaneous full-recording downloads.
    const task = queue.then(async () => {
        // A queued card may have disappeared or its signed URL may have expired.
        if (!(await canGenerate())) throw new DOMException("Obsolete spectrogram", "AbortError");

        const buffer = await decodeRecording(clip, onStatus);
        onStatus("Generating spectrogram...");
        return drawSpectrogram(buffer, clip.offsets);
    });
    queue = task.catch(() => {});
    images.set(key, task);
    task.catch(() => {
        if (images.get(key) === task) images.delete(key);
    });
    if (images.size > 40) images.delete(images.keys().next().value);
    return task;
}

export function localSpectrogram(clip, ensureFresh) {
    const wrapper = document.createElement("div");
    wrapper.className = "media-image";
    const status = document.createElement("p");
    status.className = "media-loading";
    status.setAttribute("role", "status");
    status.textContent = "Spectrogram queued...";
    wrapper.append(status);

    async function generate() {
        if (!wrapper.isConnected) return;
        try {
            if (!(await ensureFresh())) return;
            const url = await generateSpectrogram(
                clip,
                (message) => { status.textContent = message; },
                () => wrapper.isConnected && ensureFresh(),
            );
            if (!wrapper.isConnected) return;
            const image = document.createElement("img");
            image.alt = "Locally generated candidate spectrogram";
            image.className = "generated-spectrogram";
            image.src = url;
            wrapper.replaceChildren(image);
        } catch {
            if (!wrapper.isConnected) return;
            status.textContent = "Could not generate spectrogram. Audio playback is still available.";
            const retry = document.createElement("button");
            retry.type = "button";
            retry.textContent = "Retry spectrogram";
            retry.addEventListener("click", () => {
                wrapper.replaceChildren(status);
                generate();
            });
            wrapper.replaceChildren(status, retry);
        }
    }

    whenVisible(wrapper, generate);
    return wrapper;
}
