import { element } from "./annotation-dom.mjs";

const $ = (selector) => document.querySelector(selector);

function image(url, alt, loading = "lazy") {
    const wrapper = element("div", null, "media-image");
    const indicator = loadingIndicator("Loading image...");
    const img = element("img");
    img.alt = alt;
    img.loading = loading;

    img.addEventListener("load", () => indicator.remove());
    img.addEventListener("error", () => {
        indicator.remove();
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
    if (clip.spectrogramUrl) {
        if (isTarget) {
            const details = element("details", null, "reference-spectrogram");
            details.open = !mobileLayout.matches;
            details.append(
                element("summary", "Target spectrogram"),
                image(clip.spectrogramUrl, `Spectrogram: ${label}`, "eager"),
            );
            wrapper.append(details);
        } else
            wrapper.append(
                image(clip.spectrogramUrl, `Spectrogram: ${label}`),
            );
    }
    const audio = element("audio");
    const indicator = loadingIndicator("Loading audio...");
    indicator.hidden = !isTarget;

    const showLoading = () => { indicator.hidden = false; };
    const hideLoading = () => { indicator.hidden = true; };
    audio.addEventListener("loadstart", () => {
        if (audio.preload !== "none") showLoading();
    });
    audio.addEventListener("waiting", showLoading);
    audio.addEventListener("seeking", showLoading);
    audio.addEventListener("canplay", hideLoading);
    audio.addEventListener("playing", hideLoading);
    audio.addEventListener("seeked", hideLoading);
    audio.addEventListener("suspend", () => {
        if (audio.readyState >= 3 || audio.preload === "none") hideLoading();
    });
    audio.controls = true;
    audio.preload = isTarget ? "auto" : "none";
    audio.setAttribute("aria-label", label);
    audio.addEventListener("error", () => {
        hideLoading();
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

    if (species.clips.length > 1) card.append(navigation);
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

