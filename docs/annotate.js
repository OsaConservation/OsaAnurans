import { CandidateQueue } from "./candidate-queue.mjs?v=weighted-sampling-1";
import { createSubmission } from "./annotation-submission.mjs";
import { createAnnotationSummary } from "./annotation-summary.mjs?v=query-efficiency-1";
import { installSpectrogramPreview } from "./spectrogram-preview.mjs";
﻿import { element } from "./annotation-dom.mjs";
import { renderReferences, sound } from "./reference-media.mjs?v=reference-spectrogram-1";
import { candidatePlayer } from "./candidate-player.mjs?v=reference-spectrogram-1";
import { CandidateCursorError, createProvider } from "./annotation-data.mjs?v=weighted-sampling-1";

// Page controls. All reads and saves go through the configured data provider.

const $ = (selector) => document.querySelector(selector);
const dropdown = $("#scientific");
const saveButton = $("#sendButton");
const status = $("#status");
const retry = $("#retry");
const nextButton = element("button", "Next batch");
nextButton.type = "button";
document.querySelector(".save-bar").append(nextButton);
const cloudMode = window.annotationConfig.provider === "server";
const batchSize = $("#batch-size");
const batchSizeValue = $("#batch-size-value");
$("#batch-size-control").hidden = !cloudMode;
let requestVersion = 0;
let candidateRequest;
let mediaRefreshAttempted = false;
let candidateRefresh;
let candidateQueue;
// Keep each species' visible batch and remaining queue until the page reloads.
const speciesBatches = new Map();
// Reload creates a new ordering; species switches, slider changes, and retries reuse it.
const samplingSeed = crypto.randomUUID();
let retryAction;
let referenceRefresh;

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

// Answers remain drafts until a matching submission receipt is received.
// Keys are clip/window IDs; values are "yes", "no", or "unsure".

const answers = new Map();

let provider;
let speciesId;
const speciesById = new Map();
let batch;
let loading = false;
let saving = false;
let pendingSubmission = null;
let sessionSubmittedCount = 0;
const submittedAnswers = new Map();
const annotatorName = $("#annotator-name");
const workspace = $("#annotator-workspace");
const spectrogramPreview = installSpectrogramPreview(workspace);
const loginPanel = $("#annotator-login");
let annotatorId;
let resumeRead;
let annotationSummary;

// Each page load starts a fresh session. Remove drafts from older versions too.
function clearStoredAnnotationState() {
    for (const storageName of ["localStorage", "sessionStorage"]) {
        try {
            const storage = window[storageName];
            const keys = Object.keys(storage).filter((key) =>
                key === "osa-reviewer-id" || key.startsWith("osa-annotation-draft") ||
                key.startsWith("annotation-catalog-"));
            keys.forEach((key) => storage.removeItem(key));
        } catch { /* Storage restrictions must not prevent annotation. */ }
    }
}

clearStoredAnnotationState();
annotatorName.value = "";
if (cloudMode) {
    annotatorId = crypto.randomUUID();
    provider = createProvider(window.annotationConfig);
    provider.api.addEventListener("logout", lockPage);
}

function lockPage() {
    spectrogramPreview.close();
    requestVersion++;
    candidateRequest?.abort();
    loading = false;
    workspace.hidden = true;
    loginPanel.hidden = false;
    document.querySelectorAll("audio").forEach((audio) => audio.pause());
    $("#login-error").textContent = "Enter the annotator password to continue. Your answers are preserved.";
}

function canReview() {
    if (!cloudMode) return true;
    if (provider.api.authenticated) return true;
    if (!workspace.hidden) provider.api.logout();
    return false;
}

function batchHasChanges() {
    return batch?.clips.some((clip) => answers.has(clip.id) && answers.get(clip.id) !== submittedAnswers.get(clip.id));
}


// Warn before leaving a page with draft answers.

function dirty() {
    return Boolean(pendingSubmission) || [...answers].some(([id, value]) => value !== submittedAnswers.get(id));
}

function updateProgress() {
    window.removeEventListener("beforeunload", warnBeforeLeaving);
    if (dirty()) window.addEventListener("beforeunload", warnBeforeLeaving);

    $("#progress").textContent = `${answers.size} of ${batch?.clips.length || 0} answered; answers stay in this page only`;
    saveButton.disabled = loading || saving || !canReview() || (!pendingSubmission && !batchHasChanges());
    saveButton.hidden = !cloudMode;
    nextButton.hidden = !cloudMode;
    nextButton.disabled = loading || saving || !canReview() || Boolean(pendingSubmission) || batchHasChanges() || !batch?.hasNext;
    batchSize.disabled = loading || saving || Boolean(pendingSubmission);
    if (cloudMode) {
        $("#progress").textContent = `${sessionSubmittedCount} answers submitted in this session`;
    }
    dropdown.disabled = loading || saving || Boolean(pendingSubmission) || !dropdown.options.length;
    annotationSummary?.updateLocks();
    document.querySelectorAll(".answer-options input").forEach((input) => {
        input.disabled = loading || saving || Boolean(pendingSubmission);
    });
    saveButton.textContent = saving ? "Submitting..." : pendingSubmission ? "Retry submission" : "Submit answers";
    annotatorName.disabled = saving || Boolean(pendingSubmission);
    $("#edit-pending").hidden = !pendingSubmission;
    $("#edit-pending").disabled = saving;
}

// Native radio groups support mouse, touch, and keyboard interaction.

function renderCandidates() {
    const renderedBatch = batch;
    const list = $(".results-list");
    list.replaceChildren();
    batch.clips.forEach((clip, i) => {
        const li = element("li");
        li.append(
            element("h3", `Clip ${i + 1}, ID: ${clip.windowId ?? clip.id}`),
            cloudMode ? candidatePlayer(clip, {
                ensureFresh: () => batch === renderedBatch && ensureCandidateMediaFresh(),
                onFailure: () => {
                    if (batch === renderedBatch) refreshFailedCandidateMedia();
                },
            }) : sound(clip, `Clip ${i + 1}`),
        );
        li.append(answerChoices(clip, i));
        list.append(li);
    });
    if (!batch.clips.length)
        list.append(
            element("li", "No clips are waiting for review for this species."),
        );
}

function answerChoices(clip, index) {
    const choices = element("fieldset", null, "answer-options");
    const species = speciesById.get(speciesId);
    const name = species?.commonName || species?.scientificName || speciesId;
    choices.append(element("legend", `Can you hear ${name}?`));
    for (const [value, text] of [
        ["yes", "Yes"],
        ["no", "No"],
        ["unsure", "Unsure"],
    ]) {
        const label = element("label"),
            input = element("input");
        input.type = "radio";
        input.name = `clip-${index}`;
        input.value = value;
        input.checked = answers.get(clip.id) === value;
        input.addEventListener("change", () => {
            answers.set(clip.id, value);
            status.textContent = "";
            updateProgress();
        });
        label.append(input, document.createTextNode(text));
        choices.append(label);
    }
    return choices;
}

// Only the latest selection may replace the visible batch.

function showReferenceLoading(label) {
    const notice = $("#target-reference-loading");
    notice.hidden = !label;
    notice.textContent = label || "";
    $("#frog-card").hidden = Boolean(label);
    $(".target-reference").setAttribute("aria-busy", String(Boolean(label)));
}

async function loadSpecies(id, count = Number(batchSize.value), { restart = false } = {}) {
    if (!canReview() || saving || pendingSubmission) return;
    const version = ++requestVersion;
    candidateRequest?.abort();
    candidateRequest = new AbortController();
    loading = true;
    retry.hidden = true;
    retry.textContent = "Retry";
    status.textContent = "Loading references and clips...";
    showReferenceLoading(`Loading target reference for ${speciesById.get(id)?.scientificName || id}…`);
    document.querySelectorAll("audio").forEach((audio) => audio.pause());
    speciesId = id;
    retryAction = () => loadSpecies(id, count, { restart });
    resumeRead = retryAction;
    if (!cloudMode) {
        answers.clear();
    }
    updateProgress();

    const saved = cloudMode && !restart ? speciesBatches.get(id) : null;
    const queue = saved?.queue ?? (cloudMode ? new CandidateQueue(provider, id, samplingSeed) : null);
    if (saved) count = saved.batch.count;
    try {
        const [references, nextBatch] = await Promise.all([
            provider.getReferenceSounds(id),
            saved ? restoreCandidateBatch(saved.batch, candidateRequest.signal)
                : cloudMode ? queue.next(count, candidateRequest.signal) : provider.getCandidates(id),
        ]);
        if (version !== requestVersion) return;

        batch = nextBatch;
        candidateQueue = queue;
        rememberSpeciesBatch();
        if (cloudMode) resumeRead = refreshCurrentBatch;
        batchSize.value = count;
        batchSizeValue.value = count;
        mediaRefreshAttempted = false;
        renderReferences(references);
        renderCandidates();
        showBatchStatus();
    } catch (error) {
        if (version !== requestVersion) return;
        if (error instanceof CandidateCursorError) {
            retryAction = () => loadSpecies(id, count, { restart: true });
            retry.textContent = "Restart batch";
        }
        if (cloudMode && batch?.speciesId) {
            speciesId = batch.speciesId;
            dropdown.value = speciesId;
            batchSize.value = batch.count;
            batchSizeValue.value = batch.count;
        }
        status.textContent = `Could not load this species: ${error.message}`;
        retry.hidden = false;
    } finally {
        if (version === requestVersion) {
            showReferenceLoading(null);
            loading = false;
            updateProgress();
        }
    }
}

function rememberSpeciesBatch() {
    if (cloudMode) speciesBatches.set(batch.speciesId, { batch, queue: candidateQueue });
}

async function restoreCandidateBatch(saved, signal) {
    if (!saved.clips.length || Date.parse(saved.urlsExpireAt) - Date.now() >= 60_000) return saved;

    // Refresh the same questions; never advance a saved queue to renew media URLs.
    return provider.refreshCandidateMedia(saved, signal);
}

function showBatchStatus() {
    if (!cloudMode) { status.textContent = ""; return; }

    status.textContent = !batch.hasNext
        ? "This is the last batch for this species. Submit any remaining answers."
        : "Submit answered clips before moving to the next batch.";
}

async function loadCandidateBatch({ refresh = false } = {}) {
    if (!batch || loading || saving || !canReview()) return false;
    const current = batch;
    retryAction = () => loadCandidateBatch({ refresh });
    resumeRead = retryAction;
    const version = ++requestVersion;
    loading = true;
    retry.hidden = true;
    retry.textContent = "Retry";
    status.textContent = refresh ? "Refreshing clip media..." : "Loading next batch...";
    updateProgress();

    try {
        candidateRequest?.abort();
        candidateRequest = new AbortController();
        const nextBatch = refresh
            ? await provider.refreshCandidateMedia(current, candidateRequest.signal)
            : await candidateQueue.next(current.count, candidateRequest.signal);
        if (version !== requestVersion) return false;

        document.querySelectorAll(".results-list audio").forEach((audio) => audio.pause());
        batch = nextBatch;
        rememberSpeciesBatch();
        resumeRead = refreshCurrentBatch;
        if (!refresh) mediaRefreshAttempted = false;
        renderCandidates();
        showBatchStatus();
        return true;
    } catch (error) {
        if (version !== requestVersion) return false;
        if (error instanceof CandidateCursorError) {
            retryAction = () => loadSpecies(current.speciesId, current.count, { restart: true });
            retry.textContent = "Restart batch";
        }
        if (error.code === "CANDIDATES_REMOVED") {
            retry.textContent = "Restart species";
            retryAction = () => {
                if (batchHasChanges() || pendingSubmission) {
                    status.textContent = "Submit pending answers before restarting this species.";
                    return;
                }
                loadSpecies(current.speciesId, current.count, { restart: true });
            };
        }
        status.textContent = `Could not load candidates: ${error.message}. Your answers are preserved.`;
        retry.hidden = false;
        return false;
    } finally {
        if (version === requestVersion) {
            loading = false;
            updateProgress();
        }
    }
}

async function ensureCandidateMediaFresh() {
    if (!batch?.clips.length || !canReview()) return false;
    if (Date.parse(batch.urlsExpireAt) - Date.now() >= 60_000) return true;

    await refreshCurrentBatch();
    return false;
}

function refreshCurrentBatch() {
    if (candidateRefresh) return candidateRefresh;

    candidateRefresh = loadCandidateBatch({ refresh: true })
        .finally(() => { candidateRefresh = null; });
    return candidateRefresh;
}

function refreshFailedCandidateMedia() {
    if (mediaRefreshAttempted) {
        status.textContent = "Media unavailable after refreshing. Use Retry loading to try again.";
        retry.hidden = false;
        retryAction = refreshCurrentBatch;
        return;
    }

    mediaRefreshAttempted = true;
    refreshCurrentBatch();
}

nextButton.addEventListener("click", () => {
    if (!batch?.hasNext) return;
    loadCandidateBatch();
});

batchSize.addEventListener("input", () => {
    batchSizeValue.value = batchSize.value;
});

batchSize.addEventListener("change", () => {
    if (!cloudMode || !speciesId || loading || saving || pendingSubmission) return;

    loadSpecies(speciesId, Number(batchSize.value), { restart: true });
});

window.addEventListener("pageshow", () => {
    if (cloudMode && batch && canReview()) ensureCandidateMediaFresh();
});
document.addEventListener("visibilitychange", () => {
    if (!document.hidden && cloudMode && batch && canReview()) ensureCandidateMediaFresh();
});

// User actions: species selection, saving, navigation protection, and loading retries.

dropdown.addEventListener("change", () => {
    if (saving || pendingSubmission) return;
    if (!cloudMode && dirty() && !window.confirm("Discard unsaved answers and change species?")) {
        dropdown.value = speciesId;
        return;
    }
    loadSpecies(dropdown.value);
});
function warnBeforeLeaving(event) {
    if (!dirty()) return;

    event.preventDefault();
    event.returnValue = "";
}

async function refreshReferenceMedia() {
    if (!canReview()) return;
    if (referenceRefresh) return referenceRefresh;
    const selectedSpecies = speciesId;
    showReferenceLoading("Refreshing target reference media…");

    referenceRefresh = provider.getReferenceSounds(selectedSpecies).then((references) => {
        if (speciesId !== selectedSpecies) return;
        renderReferences(references);
        status.textContent = "Reference links refreshed. Press Play again.";
    }).catch((error) => {
        if (speciesId !== selectedSpecies) return;
        status.textContent = `Could not refresh reference media: ${error.message}`;
        retryAction = refreshReferenceMedia;
        retry.hidden = false;
    }).finally(() => {
        if (speciesId === selectedSpecies) showReferenceLoading(null);
        referenceRefresh = null;
    });
    return referenceRefresh;
}

// Play events do not bubble; capture them to prevent overlapping reference/test audio.

document.addEventListener(
    "play",
    (event) => {
        if (event.target.tagName !== "AUDIO") return;
        if (cloudMode && event.target.closest(".reference-panel") && !provider.referencesAreFresh()) {
            event.target.pause();
            refreshReferenceMedia();
            return;
        }
        document.querySelectorAll("audio").forEach((audio) => {
                if (audio !== event.target) audio.pause();
            });
    },
    true,
);
retry.addEventListener("click", () => {
    mediaRefreshAttempted = false;
    if (retryAction) return retryAction();

    initialize();
});

// Startup is also the retry path when configuration or species loading fails.

async function initialize() {
    if (!canReview()) return;
    loading = true;
    retry.hidden = true;
    status.textContent = "Loading species...";
    updateProgress();
    try {
        provider ||= createProvider(window.annotationConfig);
        $("#mode-note").textContent =
            window.annotationConfig.provider === "demo"
                ? "Demo mode - Sample clips are reused for every species. Answers stay in this page only."
                : "Cloud review - Submit your answers to save them.";
        const species = await provider.listSpecies();
        if (!species.length) throw new Error("No species are available.");
        speciesById.clear();
        species.forEach((item) => speciesById.set(item.id, item));
        dropdown.replaceChildren(
            ...species.map((item) => {
                const option = element("option", item.scientificName);
                option.value = item.id;
                return option;
            }),
        );
        if (speciesId && speciesById.has(speciesId)) dropdown.value = speciesId;
        if (cloudMode) {
            annotationSummary ||= createAnnotationSummary({
                provider,
                available: (id) => speciesById.has(id),
                locked: () => loading || saving || Boolean(pendingSubmission) || !provider.api.authenticated,
                select: (id) => {
                    dropdown.value = id;
                    dropdown.dispatchEvent(new Event("change"));
                },
            });
        }
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

async function submitAnswers() {
    if (!cloudMode || saving || loading || !canReview()) return;
    saving = true;
    try {
        if (!pendingSubmission) {
            const changed = new Map([...answers].filter(([id, value]) => submittedAnswers.get(id) !== value));
            pendingSubmission = createSubmission(batch.clips, changed, annotatorId, annotatorName.value);
        }
        // Keep the exact payload in memory for retries within this page.
        updateProgress();
        status.textContent = "Submitting answers...";
        await provider.saveAnnotations(pendingSubmission);
        sessionSubmittedCount += pendingSubmission.annotations.length;
        const labels = { present: "yes", absent: "no", unsure: "unsure" };
        for (const item of pendingSubmission.annotations) submittedAnswers.set(item.candidate_id, labels[item.response]);
        pendingSubmission = null;
        annotationSummary?.markStale();
        status.textContent = batch?.hasNext
            ? "Answers submitted successfully. Select Next batch to continue."
            : "Answers submitted successfully. No more candidates for this species. Select another species to continue.";

    } catch (error) {
        annotationSummary?.markStale();
        status.textContent = `${error.message} Pending answers are retained. A manual retry after a network error may create duplicates.`;
    } finally {
        saving = false;
        updateProgress();
    }
}

saveButton.addEventListener("click", submitAnswers);
$("#edit-pending").addEventListener("click", () => {
    if (saving || !pendingSubmission) return;
    if (!window.confirm("Stop retrying this payload and edit answers? If the previous request succeeded, a new submission may duplicate those answers.")) return;
    pendingSubmission = null;
    updateProgress();
});
$("#logout-button").addEventListener("click", () => provider.api.logout());
function setPasswordVisible(visible) {
    $("#annotator-password").type = visible ? "text" : "password";
    const label = visible ? "Hide password" : "Show password";
    $("#toggle-password").setAttribute("aria-label", label);
    $("#toggle-password").title = label;
    $("#password-eye-slash").toggleAttribute("hidden", !visible);
}
$("#toggle-password").addEventListener("click", () => {
    setPasswordVisible($("#annotator-password").type === "password");
});
$("#login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("#login-button");
    if (button.disabled) return;
    button.disabled = true;
    const password = $("#annotator-password");
    try {
        await provider.api.login(password.value);
        loginPanel.hidden = true;
        workspace.hidden = false;
        $("#login-error").textContent = "";
        if (batch) {
            updateProgress();
            if (!pendingSubmission && resumeRead) await resumeRead();
        } else await initialize();
    } catch (error) {
        $("#login-error").textContent = error.message;
    } finally {
        password.value = "";
        setPasswordVisible(false);
        button.disabled = false;
    }
});

loginPanel.hidden = !cloudMode;
workspace.hidden = cloudMode;
$("#logout-button").hidden = !cloudMode;
$("#annotator-controls").hidden = !cloudMode;
annotatorName.hidden = !cloudMode;
if (!cloudMode) initialize();
