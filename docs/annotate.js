import { createProvider, buildSubmission } from "./annotation-data.mjs";

// Page controls. All reads and saves go through the configured data provider.

const $ = (selector) => document.querySelector(selector);
const dropdown = $("#scientific");
const saveButton = $("#sendButton");
const status = $("#status");
const retry = $("#retry");

// Compare current answers with acknowledged answers to detect unsaved edits.
// Keys are recording/window IDs; values are "yes", "no", or "unsure".

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

// Only new or changed answers count as unsaved; unanswered recordings are skipped.

function dirty() {
    return [...answers].some(([id, label]) => savedAnswers.get(id) !== label);
}

function updateProgress() {
    const changed = [...answers].filter(
        ([id, label]) => savedAnswers.get(id) !== label,
    ).length;
    $("#progress").textContent =
        `${answers.size} of ${batch?.recordings.length || 0} answered · ${changed} unsaved`;
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
function sound(example, label, isTarget = false) {
    const wrapper = element("div", null, "sound-example");
    if (example.spectrogramUrl) {
        if (isTarget) {
            const details = element("details", null, "reference-spectrogram");
            details.open = !mobileLayout.matches;
            details.append(
                element("summary", "Target spectrogram"),
                image(example.spectrogramUrl, `Spectrogram: ${label}`),
            );
            wrapper.append(details);
        } else
            wrapper.append(
                image(example.spectrogramUrl, `Spectrogram: ${label}`),
            );
    }
    const audio = element("audio");
    audio.src = example.audioUrl;
    audio.controls = true;
    audio.preload = "none";
    audio.setAttribute("aria-label", label);
    audio.addEventListener("error", () => {
        if (!wrapper.querySelector(".audio-error"))
            wrapper.append(
                element(
                    "p",
                    "Audio unavailable. Please try another recording.",
                    "media-error audio-error",
                ),
            );
    });
    wrapper.append(audio);
    return wrapper;
}

// The target shows one selectable example; other frogs use an example carousel.

function referenceCard(species, isTarget = false) {
    const card = element("article", null, "frog-card");
    card.append(
        element("h3", species.scientificName),
        element("p", species.commonName),
    );
    if (species.imageUrl)
        card.append(image(species.imageUrl, species.scientificName));
    const examples = element("div", null, "reference-examples");
    if (isTarget && species.examples.length) {
        const select = element("select");
        select.setAttribute("aria-label", "Target reference example");
        species.examples.forEach((_example, i) => {
            const option = element(
                "option",
                `Reference ${i + 1} of ${species.examples.length}`,
            );
            option.value = i;
            select.append(option);
        });
        const showExample = () => {
            // Pause the outgoing example before removing its audio element.
            examples
                .querySelectorAll("audio")
                .forEach((audio) => audio.pause());
            examples.replaceChildren(
                sound(
                    species.examples[Number(select.value)],
                    `${species.scientificName}, example ${Number(select.value) + 1}`,
                    true,
                ),
            );
        };
        select.addEventListener("change", showExample);
        if (species.examples.length > 1) card.append(select);
        showExample();
    } else
        species.examples.forEach((example, i) =>
            examples.append(
                sound(example, `${species.scientificName}, example ${i + 1}`),
            ),
        );
    if (!species.examples.length)
        examples.append(element("p", "No reference audio available."));
    card.append(examples);
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

function renderRecordings() {
    const list = $(".results-list");
    list.replaceChildren();
    batch.recordings.forEach((recording, i) => {
        const li = element("li");
        li.append(
            element("h3", `Recording ${i + 1}`),
            sound(recording, `Recording ${i + 1}`),
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
            input.name = `recording-${i}`;
            input.value = value;
            input.addEventListener("change", () => {
                answers.set(recording.id, value);
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
    if (!batch.recordings.length)
        list.append(
            element("li", "No recordings are waiting for this species."),
        );
}

// A species change starts a fresh batch, after the change handler confirms any edits.

async function loadSpecies(id) {
    loading = true;
    retry.hidden = true;
    status.textContent = "Loading references and recordings…";
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
            provider.getRecordings(id),
        ]);
        if (
            new Set(nextBatch.recordings.map((item) => item.id)).size !==
            nextBatch.recordings.length
        )
            throw new Error("Recording IDs must be unique within a batch.");
        batch = nextBatch;
        renderReferences(references);
        renderRecordings();
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
        batch.recordings,
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
                ? "Demo mode · Sample recordings are reused for every species. Saves stay in this browser."
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
