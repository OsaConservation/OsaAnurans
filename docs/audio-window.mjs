/** Limit a full recording to one review interval. Boundaries are approximate. */
export function limitPlayback(audio, start, end, clock = globalThis) {
    if (
        !Number.isFinite(start) ||
        !Number.isFinite(end) ||
        start < 0 ||
        end <= start
    )
        throw new Error("Invalid audio window.");

    let stopTimer;

    // Cancel on pause or buffering; elapsed wall time is not playback time.
    function cancelStop() {
        clock.clearTimeout(stopTimer);
    }

    function stop() {
        cancelStop();
        audio.pause();
        audio.currentTime = end;
    }

    function scheduleStop() {
        cancelStop();
        if (audio.paused || audio.seeking || audio.readyState < 2) return;
        const remaining = (end - audio.currentTime) / audio.playbackRate;
        if (remaining <= 0) stop();
        else stopTimer = clock.setTimeout(stop, remaining * 1000);
    }

    function clamp() {
        if (audio.currentTime < start) audio.currentTime = start;
        else if (audio.currentTime > end) stop();
    }

    // Seek once metadata is available and clamp later user/browser seeks.
    audio.addEventListener("loadedmetadata", () => {
        audio.currentTime = start;
    });
    audio.addEventListener("play", () => {
        if (audio.currentTime < start || audio.currentTime >= end)
            audio.currentTime = start;
    });
    audio.addEventListener("seeking", () => {
        cancelStop();
        clamp();
    });
    audio.addEventListener("timeupdate", () => {
        clamp();
        if (!audio.paused && audio.currentTime >= end) stop();
    });

    // Schedule only while playback is progressing; buffering must not consume the window.
    for (const event of ["playing", "seeked", "ratechange"])
        audio.addEventListener(event, scheduleStop);
    for (const event of ["pause", "waiting", "ended", "emptied", "error"])
        audio.addEventListener(event, cancelStop);
}
