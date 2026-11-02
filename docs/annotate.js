import { createProvider, buildSubmission } from "./annotation-data.mjs";

// Page controls. All reads and saves go through the configured data provider.

const $ = (selector) => document.querySelector(selector);
const dropdown = $("#scientific");
const saveButton = $("#sendButton");
const status = $("#status");
const retry = $("#retry");

// Leave room below the desktop reference panel for the sticky save controls.
// Observe wrapping and text changes as well as viewport resizing.
const saveBar = $(".save-bar");
const updateSaveBarHeight = () => {
    document.documentElement.style.setProperty(
        "--save-bar-height",
        `${saveBar.getBoundingClientRect().height}px`,
    );
};
updateSaveBarHeight();
new ResizeObserver(updateSaveBarHeight).observe(saveBar);

// Compare current answers with acknowledged answers to detect unsaved edits.
// Keys are clip/window IDs; values are "yes", "no", or "unsure".

const answers = new Map();
const savedAnswers = new Map();

let provider;
let speciesId;
let batch;
let loading = false;
let saving = false;
let pendingSubmission;

// Use textContent so species names and server text are never interpreted as HTML.

function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (className) node.className = className;
    return node;
}

// Only new or changed answers count as unsaved; unanswered clips are skipped.

function dirty() {
    return [...answers].some(([id, label]) => savedAnswers.get(id) !== label);
}

function updateProgress() {
    const changed = [...answers].filter(
        ([id, label]) => savedAnswers.get(id) !== label,
    ).length;
    $("#progress").textContent =
        `${answers.size} of ${batch?.clips.length || 0} answered · ${changed} unsaved`;
    saveButton.disabled = loading || saving || !changed;
    dropdown.disabled = loading || saving || !dropdown.options.length;
    document.querySelectorAll(".answer-options input").forEach((input) => {
        input.disabled = saving;
    });
    saveButton.textContent = saving ? "Saving…" : "Save annotations";
}

function image(url, alt) {
    const img = element("img");
    img.src = url;
    img.alt = alt;
    img.loading = "lazy";
    img.addEventListener("error", () =>
        img.replaceWith(element("p", `${alt} unavailable.`, "media-error")),
    );
    return img;
}

// Keep this breakpoint aligned with CSS. Crossing it resets the spectrogram to
// expanded on desktop or collapsed on mobile; users can then toggle it themselves.

const mobileLayout = window.matchMedia("(max-width: 760px)");
mobileLayout.addEventListener("change", () => {
    document.querySelectorAll(".reference-spectrogram").forEach((details) => {
        details.open = !mobileLayout.matches;
    });
});
function sound(clip, label, isTarget = false) {
    const wrapper = element("div", null, "audio-clip");
    if (clip.spectrogramUrl) {
        if (isTarget) {
            const details = element("details", null, "reference-spectrogram");
            details.open = !mobileLayout.matches;
            details.append(
                element("summary", "Target spectrogram"),
                image(clip.spectrogramUrl, `Spectrogram: ${label}`),
            );
            wrapper.append(details);
        } else
            wrapper.append(
                image(clip.spectrogramUrl, `Spectrogram: ${label}`),
            );
    }
    const audio = element("audio");
    audio.src = clip.audioUrl;
    audio.controls = true;
    audio.preload = "none";
    audio.setAttribute("aria-label", label);
    audio.addEventListener("error", () => {
        if (!wrapper.querySelector(".audio-error"))
            wrapper.append(
                element(
                    "p",
                    "Audio unavailable. Please try another clip.",
                    "media-error audio-error",
                ),
            );
    });
    wrapper.append(audio);
    return wrapper;
}

// The target shows one selectable clip; other frogs use a clip carousel.

function referenceCard(species, isTarget = false) {
    const card = element("article", null, "frog-card");
    card.append(
        element("h3", species.scientificName),
        element("p", species.commonName),
    );
    if (species.imageUrl)
        card.append(image(species.imageUrl, species.scientificName));
    const clips = element("div", null, "reference-clips");
    if (isTarget && species.clips.length) {
        const select = element("select");
        select.setAttribute("aria-label", "Target reference clip");
        species.clips.forEach((_clip, i) => {
            const option = element(
                "option",
                `Reference ${i + 1} of ${species.clips.length}`,
            );
            option.value = i;
            select.append(option);
        });
        const showClip = () => {
            // Pause the outgoing clip before removing its audio element.
            clips
                .querySelectorAll("audio")
                .forEach((audio) => audio.pause());
            clips.replaceChildren(
                sound(
                    species.clips[Number(select.value)],
                    `${species.scientificName}, clip ${Number(select.value) + 1}`,
                    true,
                ),
            );
        };
        select.addEventListener("change", showClip);
        if (species.clips.length > 1) card.append(select);
        showClip();
    } else
        species.clips.forEach((clip, i) =>
            clips.append(
                sound(clip, `${species.scientificName}, clip ${i + 1}`),
            ),
        );
    if (!species.clips.length)
        clips.append(element("p", "No reference audio available."));
    card.append(clips);
    return card;
}

function renderReferences(references) {
    $("#frog-card").replaceChildren(referenceCard(references.target, true));
    $(".frog-carousel").replaceChildren(
        ...references.comparisons.map((species) => {
            const li = element("li");
            li.append(referenceCard(species));
            return li;
        }),
    );
}

// Native radio groups support mouse, touch, and keyboard interaction.

function renderCandidates() {
    const list = $(".results-list");
    list.replaceChildren();
    batch.clips.forEach((clip, i) => {
        const li = element("li");
        li.append(
            element("h3", `Clip ${i + 1}`),
            sound(clip, `Clip ${i + 1}`),
        );
        const choices = element("fieldset", null, "answer-options");
        choices.append(element("legend", `Can you hear ${speciesId}?`));
        for (const [value, text] of [
            ["yes", "Yes"],
            ["no", "No"],
            ["unsure", "Unsure"],
        ]) {
            const label = element("label"),
                input = element("input");
            input.type = "radio";
            input.name = `clip-${i}`;
            input.value = value;
            input.addEventListener("change", () => {
                answers.set(clip.id, value);
                pendingSubmission = null;
                status.textContent = "";
                updateProgress();
            });
            label.append(input, document.createTextNode(text));
            choices.append(label);
        }
        li.append(choices);
        list.append(li);
    });
    if (!batch.clips.length)
        list.append(
            element("li", "No clips are waiting for review for this species."),
        );
}

// A species change starts a fresh batch, after the change handler confirms any edits.

async function loadSpecies(id) {
    loading = true;
    retry.hidden = true;
    status.textContent = "Loading references and clips…";
    // Clear old media immediately so users cannot compare against the wrong target.
    document.querySelectorAll("audio").forEach((audio) => audio.pause());
    $("#frog-card").replaceChildren();
    $(".frog-carousel").replaceChildren();
    $(".results-list").replaceChildren();
    answers.clear();
    savedAnswers.clear();
    batch = null;
    pendingSubmission = null;
    speciesId = id;
    updateProgress();
    try {
        const [references, nextBatch] = await Promise.all([
            provider.getReferenceSounds(id),
            provider.getCandidates(id),
        ]);
        if (
            new Set(nextBatch.clips.map((item) => item.id)).size !==
            nextBatch.clips.length
        )
            throw new Error("Clip window IDs must be unique within a batch.");
        batch = nextBatch;
        renderReferences(references);
        renderCandidates();
        status.textContent = "";
    } catch (error) {
        status.textContent = `Could not load this species: ${error.message}`;
        retry.hidden = false;
    } finally {
        loading = false;
        updateProgress();
    }
}

// User actions: species selection, saving, navigation protection, and loading retries.

dropdown.addEventListener("change", () => {
    if (
        dirty() &&
        !window.confirm("Discard unsaved answers and change species?")
    ) {
        dropdown.value = speciesId;
        return;
    }
    loadSpecies(dropdown.value);
});
saveButton.addEventListener("click", async () => {
    if (saving || loading || !dirty()) return;
    // Reuse the ID after an ambiguous failure so retries cannot duplicate a batch.
    pendingSubmission ||= buildSubmission(
        speciesId,
        batch.batchId,
        batch.clips,
        new Map(
            [...answers].filter(
                ([id, label]) => savedAnswers.get(id) !== label,
            ),
        ),
        crypto.randomUUID(),
    );
    // Lock answers and species until this snapshot has finished saving.
    saving = true;
    status.textContent = "Saving annotations…";
    updateProgress();
    try {
        const result = await provider.saveAnnotations(pendingSubmission);
        if (
            result.submissionId !== pendingSubmission.submissionId ||
            result.savedCount !== pendingSubmission.annotations.length
        )
            throw new Error("The server did not confirm the complete batch.");
        // Answers become clean only after the entire submission is acknowledged.
        answers.forEach((label, id) => savedAnswers.set(id, label));
        pendingSubmission = null;
        status.textContent =
            window.annotationConfig.provider === "demo"
                ? "Saved in this browser for demo purposes. No training data was sent to a server."
                : "Annotations saved. Thank you!";
    } catch (error) {
        status.textContent = `Save failed: ${error.message} Your answers are still here; try saving again.`;
    } finally {
        saving = false;
        updateProgress();
    }
});
window.addEventListener("beforeunload", (event) => {
    if (dirty()) {
        event.preventDefault();
        event.returnValue = "";
    }
});

// Play events do not bubble; capture them to prevent overlapping reference/test audio.

document.addEventListener(
    "play",
    (event) => {
        if (event.target.tagName === "AUDIO")
            document.querySelectorAll("audio").forEach((audio) => {
                if (audio !== event.target) audio.pause();
            });
    },
    true,
);
retry.addEventListener("click", () =>
    speciesId ? loadSpecies(speciesId) : initialize(),
);

// Startup is also the retry path when configuration or species loading fails.

async function initialize() {
    loading = true;
    retry.hidden = true;
    status.textContent = "Loading species…";
    updateProgress();
    try {
        provider = createProvider(window.annotationConfig);
        $("#mode-note").textContent =
            window.annotationConfig.provider === "demo"
                ? "Demo mode · Sample clips are reused for every species. Saves stay in this browser."
                : "Answers will be saved to the annotation server.";
        const species = await provider.listSpecies();
        if (!species.length) throw new Error("No species are available.");
        dropdown.replaceChildren(
            ...species.map((item) => {
                const option = element("option", item.scientificName);
                option.value = item.id;
                return option;
            }),
        );
        await loadSpecies(dropdown.value);
    } catch (error) {
        dropdown.replaceChildren();
        status.textContent = `Could not start annotation: ${error.message}`;
        retry.hidden = false;
    } finally {
        loading = false;
        updateProgress();
    }
}

initialize();
