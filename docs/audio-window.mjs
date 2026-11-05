export function waitForMetadata(audio, timeoutMs = 30000) {
    if (audio.readyState >= 1) return Promise.resolve();

    return new Promise((resolve, reject) => {
        function finish(error) {
            clearTimeout(timeout);
            audio.removeEventListener("loadedmetadata", ready);
            audio.removeEventListener("error", failed);
            if (error) reject(error);
            else resolve();
        }
        const ready = () => finish();
        const failed = () => finish(new Error("Audio metadata could not be loaded."));
        const timeout = setTimeout(() => finish(new Error("Audio loading timed out.")), timeoutMs);
        audio.addEventListener("loadedmetadata", ready);
        audio.addEventListener("error", failed);
        audio.load();
    });
}

export function windowPosition(time, [start, end]) {
    return Math.max(0, Math.min(end - start, time - start));
}
