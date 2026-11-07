import { element } from "./annotation-dom.mjs";
import { localSpectrogram } from "./local-spectrogram.mjs?v=reference-spectrogram-1";

const $ = (selector) => document.querySelector(selector);

function image(url, alt, loading = "lazy", onState = null) {
    const wrapper = element("div", null, "media-image");
    const indicator = loadingIndicator("Loading image...");
    indicator.hidden = Boolean(onState);
    const img = element("img");
    img.alt = alt;
    img.loading = loading;

    img.addEventListener("load", () => { indicator.remove(); onState?.("ready"); });
    img.addEventListener("error", () => {
        indicator.remove();
        onState?.("error");
        img.replaceWith(element("p", `${alt} unavailable.`, "media-error"));
    });

    wrapper.append(indicator, img);
    img.src = url;
    return wrapper;
}

function loadingIndicator(text) {
    const indicator = element("p", text, "media-loading");
    indicator.setAttribute("role", "status");
    return indicator;
}

// Keep this breakpoint aligned with CSS. Crossing it resets the spectrogram to
// expanded on desktop or collapsed on mobile; users can then toggle it themselves.

const mobileLayout = window.matchMedia("(max-width: 760px)");
mobileLayout.addEventListener("change", () => {
    document.querySelectorAll(".reference-spectrogram").forEach((details) => {
        details.open = !mobileLayout.matches;
    });
});
export function sound(clip, label, isTarget = false) {
    const wrapper = element("div", null, "audio-clip");
    const state = { audio: "loading", spectrogram: "loading" };
    const feedback = element("div", null, "target-media-status");
    const feedbackTitle = element("strong");
    const feedbackDetail = element("span");
    if (isTarget) {
        feedback.setAttribute("role", "status");
        feedback.setAttribute("aria-live", "polite");
        feedback.append(feedbackTitle, feedbackDetail);
        wrapper.append(feedback);
    }
    function updateTargetState(part, value) {
        state[part] = value;
        if (!isTarget) return;
        const pending = Object.values(state).includes("loading");
        const failed = Object.values(state).includes("error");
        feedback.hidden = !pending && !failed;
        feedback.classList.toggle("is-loading", pending);
        feedback.classList.toggle("is-error", failed);
        feedbackTitle.textContent = pending ? "Loading target reference…"
            : failed ? "Target reference needs attention" : "Target reference ready";
        feedbackDetail.textContent = [
            state.audio === "ready" ? "Audio ready to play" : state.audio === "error" ? "Audio unavailable" : "Audio loading…",
            state.spectrogram === "ready" ? "Spectrogram ready"
                : state.spectrogram === "error" ? "Spectrogram unavailable" : "Spectrogram loading…",
        ].join(" · ");
        wrapper.setAttribute("aria-busy", String(pending));
    }
    updateTargetState("audio", "loading");
    const alt = `Spectrogram: ${label}`;
    const onState = (value) => updateTargetState("spectrogram", value);
    const spectrogram = clip.spectrogramUrl
        ? image(clip.spectrogramUrl, alt, isTarget ? "eager" : "lazy", isTarget ? onState : null)
        : localSpectrogram(clip, undefined, { alt, eager: isTarget, onState });
    if (isTarget) {
        const details = element("details", null, "reference-spectrogram");
        details.open = !mobileLayout.matches;
        details.append(element("summary", "Target spectrogram"), spectrogram);
        wrapper.append(details);
    } else {
        const slot = element("div", null, "reference-spectrogram-slot");
        slot.append(spectrogram);
        wrapper.append(slot);
    }
    const audio = element("audio");
    const indicator = loadingIndicator("Loading audio...");
    indicator.hidden = true;

    const showLoading = () => {
        if (isTarget) updateTargetState("audio", "loading");
        else indicator.hidden = false;
    };
    const hideLoading = () => {
        if (isTarget) updateTargetState("audio", "ready");
        indicator.hidden = true;
    };
    audio.addEventListener("loadstart", () => {
        if (audio.preload !== "none") showLoading();
    });
    audio.addEventListener("waiting", showLoading);
    audio.addEventListener("seeking", showLoading);
    audio.addEventListener("canplay", hideLoading);
    audio.addEventListener("playing", hideLoading);
    audio.addEventListener("seeked", () => { if (audio.readyState >= 3) hideLoading(); });
    audio.addEventListener("suspend", () => {
        if (audio.readyState >= 3 || audio.preload === "none") hideLoading();
    });
    audio.controls = true;
    audio.preload = isTarget ? "auto" : "none";
    audio.setAttribute("aria-label", label);
    audio.addEventListener("error", () => {
        indicator.hidden = true;
        updateTargetState("audio", "error");
        if (!wrapper.querySelector(".audio-error"))
            wrapper.append(
                element(
                    "p",
                    "Audio unavailable. Please try another clip.",
                    "media-error audio-error",
                ),
            );
    });
    wrapper.append(audio, indicator);
    audio.src = clip.audioUrl;
    return wrapper;
}

// Each species shows one reference clip with previous/next navigation.

function referenceCard(species, isTarget = false, initialIndex = 0) {
    const card = element("article", null, "frog-card");
    card.append(
        element("h3", species.scientificName),
        element("p", species.commonName),
    );
    if (species.imageUrl)
        card.append(image(species.imageUrl, species.scientificName));
    card.dataset.speciesId = species.id;
    const clips = element("div", null, "reference-clips");
    if (!species.clips.length) {
        clips.append(element("p", "No reference audio available."));
        card.append(clips);
        return card;
    }

    let index = Math.max(0, Math.min(initialIndex, species.clips.length - 1));
    const navigation = element("div", null, "reference-navigation");
    const previous = element("button", "\u2190");
    const next = element("button", "\u2192");
    const position = element("span");
    position.setAttribute("role", "status");
    previous.type = next.type = "button";
    previous.setAttribute("aria-label", `Previous reference for ${species.scientificName}`);
    next.setAttribute("aria-label", `Next reference for ${species.scientificName}`);
    navigation.append(previous, position, next);

    function showClip() {
        // Stop the outgoing clip before replacing its player.
        clips.querySelectorAll("audio").forEach((audio) => audio.pause());
        card.dataset.clipIndex = index;
        previous.disabled = index === 0;
        next.disabled = index === species.clips.length - 1;
        position.textContent = `Reference ${index + 1} of ${species.clips.length}`;
        clips.replaceChildren(sound(
            species.clips[index],
            `${species.scientificName}, clip ${index + 1}`,
            isTarget,
        ));
    }

    previous.addEventListener("click", () => {
        if (index === 0) return;
        index--;
        showClip();
    });
    next.addEventListener("click", () => {
        if (index === species.clips.length - 1) return;
        index++;
        showClip();
    });

    if (species.clips.length === 1 && !isTarget) {
        // Keep the same measured height as real controls, without keyboard stops.
        navigation.classList.add("reference-navigation-placeholder");
        navigation.setAttribute("aria-hidden", "true");
        navigation.inert = true;
    }
    if (species.clips.length > 1 || !isTarget) card.append(navigation);
    showClip();
    card.append(clips);
    return card;
}

export function renderReferences(references) {
    const selections = new Map(
        [...document.querySelectorAll(".reference-panel .frog-card")].map((card) =>
            [card.dataset.speciesId, Number(card.dataset.clipIndex) || 0],
        ),
    );
    $("#frog-card").replaceChildren(referenceCard(
        references.target, true, selections.get(references.target.id) || 0,
    ));
    $(".frog-carousel").replaceChildren(
        ...references.comparisons.filter((species) => species.clips.length > 0).map((species) => {
            const li = element("li");
            li.append(referenceCard(species, false, selections.get(species.id) || 0));
            return li;
        }),
    );
}

