import { waitForMetadata, windowPosition } from "./audio-window.mjs";
import { localSpectrogram } from "./local-spectrogram.mjs?v=reference-spectrogram-1";

export function candidatePlayer(clip, { ensureFresh, onFailure }) {
    const wrapper = document.createElement("div");
    wrapper.className = "audio-clip";

    const unavailable = document.createElement("p");
    unavailable.textContent = "Spectrogram unavailable.";
    if (clip.spectrogramUrl) {
        const image = document.createElement("img");
        image.alt = "Candidate spectrogram";
        image.loading = "lazy";
        image.src = clip.spectrogramUrl;
        image.addEventListener("error", () => {
            image.replaceWith(unavailable);
            onFailure();
        }, { once: true });
        wrapper.append(image);
    } else wrapper.append(localSpectrogram(clip, ensureFresh));

    const audio = document.createElement("audio");
    audio.preload = "none";
    const [start, end] = clip.offsets;
    const duration = end - start;

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Play";

    const seek = document.createElement("input");
    seek.type = "range";
    seek.min = 0;
    seek.max = duration;
    seek.step = 0.01;
    seek.value = 0;
    seek.setAttribute("aria-label", "Position within candidate clip");

    const progress = document.createElement("span");
    const status = document.createElement("p");
    status.setAttribute("role", "status");

    function update() {
        const elapsed = windowPosition(audio.currentTime, clip.offsets);
        seek.value = elapsed;
        progress.textContent = `${elapsed.toFixed(1)} / ${duration.toFixed(1)} s`;
        if (audio.currentTime < start) audio.currentTime = start;
        if (audio.currentTime >= end) audio.pause();
    }

    audio.addEventListener("loadedmetadata", () => { audio.currentTime = start; });
    audio.addEventListener("timeupdate", update);
    audio.addEventListener("seeking", () => {
        if (audio.currentTime < start || audio.currentTime > end) audio.currentTime = start;
    });
    audio.addEventListener("waiting", () => { status.textContent = "Loading audio..."; });
    audio.addEventListener("playing", () => {
        button.textContent = "Pause";
        status.textContent = "";
    });
    audio.addEventListener("pause", () => { button.textContent = "Play"; });
    audio.addEventListener("error", () => {
        status.textContent = "Audio unavailable. Use Retry loading to try again.";
        onFailure();
    });

    // Check the end frequently as timeupdate events can be widely spaced.
    let frame;
    let stopTimer;
    function scheduleStop() {
        clearTimeout(stopTimer);
        if (audio.paused) return;

        const remaining = Math.max(0, end - audio.currentTime);
        stopTimer = setTimeout(() => {
            audio.pause();
            audio.currentTime = end;
            update();
        }, remaining / audio.playbackRate * 1000);
    }
    audio.addEventListener("playing", scheduleStop);
    audio.addEventListener("seeked", scheduleStop);
    audio.addEventListener("ratechange", scheduleStop);
    audio.addEventListener("waiting", () => clearTimeout(stopTimer));
    audio.addEventListener("pause", () => clearTimeout(stopTimer));
    function checkEnd() {
        update();
        if (!audio.paused) frame = requestAnimationFrame(checkEnd);
    }
    audio.addEventListener("play", () => { frame = requestAnimationFrame(checkEnd); });
    audio.addEventListener("pause", () => cancelAnimationFrame(frame));

    button.addEventListener("click", async () => {
        if (!audio.paused) return audio.pause();

        button.disabled = true;
        try {
            if (!(await ensureFresh())) return;
            status.textContent = "Loading audio...";
            await waitForMetadata(audio);
            if (!wrapper.isConnected) return;
            if (!Number.isFinite(audio.duration) || end > audio.duration + 0.05) {
                throw new Error("Candidate window extends outside the recording.");
            }
            if (audio.currentTime < start || audio.currentTime >= end) audio.currentTime = start;
            await audio.play();
        } catch {
            if (!wrapper.isConnected) return;
            status.textContent = "Audio unavailable. Use Retry loading to try again.";
        } finally {
            button.disabled = false;
        }
    });
    seek.addEventListener("input", () => {
        if (audio.readyState === 0) return;
        audio.currentTime = start + Number(seek.value);
        update();
    });

    progress.textContent = `0.0 / ${duration.toFixed(1)} s`;
    wrapper.append(audio, button, seek, progress, status);
    audio.src = clip.audioUrl;
    return wrapper;
}
