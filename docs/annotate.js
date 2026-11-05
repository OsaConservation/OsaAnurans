import { createSubmission, SubmissionDraft } from "./annotation-submission.mjs";
﻿import { element } from "./annotation-dom.mjs";
import { renderReferences, sound } from "./reference-media.mjs";
import { candidatePlayer } from "./candidate-player.mjs";
import { createProvider } from "./annotation-data.mjs";

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
const submittedAnswers = new Map();
const reviewerName = $("#reviewer-name");
const workspace = $("#reviewer-workspace");
const loginPanel = $("#reviewer-login");
let restoredNavigation;
let draft;
let reviewerId;
let resumeRead;

if (cloudMode) {
    provider = createProvider(window.annotationConfig);
    try {
        reviewerId = localStorage.getItem("osa-reviewer-id") || crypto.randomUUID();
        localStorage.setItem("osa-reviewer-id", reviewerId);
        draft = new SubmissionDraft(localStorage, `osa-annotation-draft-v1:${window.annotationConfig.baseUrl}`);
        const restored = draft.read();
        if (restored) {
            pendingSubmission = restored.pending || null;
            reviewerName.value = restored.name || "";
            restoredNavigation = restored.navigation;
            for (const [id, value] of restored.answers || []) answers.set(id, value);
            for (const [id, value] of restored.submitted || []) submittedAnswers.set(id, value);
        }
    } catch {
        reviewerId ||= crypto.randomUUID();
        draft = null;
    }
    provider.api.addEventListener("logout", lockPage);
}

function persistDraft() {
    if (!cloudMode || !draft) return;
    draft.write({
        name: reviewerName.value,
        answers: [...answers],
        submitted: [...submittedAnswers],
        pending: pendingSubmission,
        navigation: batch && { speciesId: batch.speciesId, count: batch.count, cursor: batch.inputCursor },
    });
}

function lockPage() {
    requestVersion++;
    candidateRequest?.abort();
    loading = false;
    workspace.hidden = true;
    loginPanel.hidden = false;
    document.querySelectorAll("audio").forEach((audio) => audio.pause());
    $("#login-error").textContent = "Enter the reviewer password to continue. Your answers are preserved.";
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
    nextButton.disabled = loading || saving || !canReview() || Boolean(pendingSubmission) || batchHasChanges() || !batch?.nextCursor;
    batchSize.disabled = loading || saving || Boolean(pendingSubmission);
    if (cloudMode) {
        const answered = batch?.clips.filter((clip) => answers.has(clip.id)).length || 0;
        $("#progress").textContent = `${answered} of ${batch?.clips.length || 0} answered in this batch; ${answers.size} drafts in this page`;
    }
    dropdown.disabled = loading || saving || Boolean(pendingSubmission) || !dropdown.options.length;
    document.querySelectorAll(".answer-options input").forEach((input) => {
        input.disabled = loading || saving || Boolean(pendingSubmission);
    });
    saveButton.textContent = saving ? "Submitting..." : pendingSubmission ? "Retry submission" : "Submit answers";
    reviewerName.disabled = saving || Boolean(pendingSubmission);
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
            try { persistDraft(); } catch { status.textContent = "Draft storage unavailable; keep this page open."; }
            status.textContent = "";
            updateProgress();
        });
        label.append(input, document.createTextNode(text));
        choices.append(label);
    }
    return choices;
}

// Only the latest selection may replace the visible batch.

async function loadSpecies(id, count = Number(batchSize.value), cursor) {
    if (!canReview() || saving) return;
    const version = ++requestVersion;
    candidateRequest?.abort();
    candidateRequest = new AbortController();
    loading = true;
    retry.hidden = true;
    status.textContent = "Loading references and clips...";
    document.querySelectorAll("audio").forEach((audio) => audio.pause());
    speciesId = id;
    retryAction = () => loadSpecies(id, count, cursor);
    resumeRead = retryAction;
    if (!cloudMode) {
        answers.clear();
    }
    updateProgress();

    try {
        const [references, nextBatch] = await Promise.all([
            provider.getReferenceSounds(id),
            provider.getCandidates(id, {
                count,
                cursor,
                signal: candidateRequest.signal,
            }),
        ]);
        if (version !== requestVersion) return;

        batch = nextBatch;
        batchSize.value = count;
        batchSizeValue.value = count;
        mediaRefreshAttempted = false;
        renderReferences(references);
        renderCandidates();
        showBatchStatus();
        try { persistDraft(); } catch { /* Continue reviewing with an in-memory draft. */ }
    } catch (error) {
        if (version !== requestVersion) return;
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
            loading = false;
            updateProgress();
        }
    }
}

function showBatchStatus() {
    if (!cloudMode) { status.textContent = ""; return; }

    status.textContent = batch.nextCursor === null
        ? "End of queue. Submit any remaining answers."
        : "Submit answered clips before moving to the next batch.";
}

async function loadCandidateBatch(cursor, { refresh = false } = {}) {
    if (!batch || loading || saving || !canReview()) return false;
    const current = batch;
    retryAction = () => loadCandidateBatch(cursor, { refresh });
    resumeRead = retryAction;
    const version = ++requestVersion;
    loading = true;
    retry.hidden = true;
    status.textContent = refresh ? "Refreshing clip media..." : "Loading next batch...";
    updateProgress();

    try {
        const nextBatch = await provider.getCandidates(current.speciesId, {
            count: current.count,
            cursor,
        });
        if (version !== requestVersion) return false;

        document.querySelectorAll(".results-list audio").forEach((audio) => audio.pause());
        batch = nextBatch;
        if (!refresh) mediaRefreshAttempted = false;
        renderCandidates();
        showBatchStatus();
        try { persistDraft(); } catch { /* Continue reviewing with an in-memory draft. */ }
        return true;
    } catch (error) {
        if (version !== requestVersion) return false;
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
    if (!batch || !canReview()) return false;
    if (Date.parse(batch.urlsExpireAt) - Date.now() >= 60_000) return true;

    await refreshCurrentBatch();
    return false;
}

function refreshCurrentBatch() {
    if (candidateRefresh) return candidateRefresh;

    candidateRefresh = loadCandidateBatch(batch.inputCursor, { refresh: true })
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
    if (batch?.nextCursor == null) return;
    loadCandidateBatch(batch.nextCursor);
});

batchSize.addEventListener("input", () => {
    batchSizeValue.value = batchSize.value;
});

batchSize.addEventListener("change", () => {
    if (!cloudMode || !speciesId || loading || saving || pendingSubmission) return;

    loadSpecies(speciesId);
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

    referenceRefresh = provider.getReferenceSounds(selectedSpecies).then((references) => {
        if (speciesId !== selectedSpecies) return;
        renderReferences(references);
        status.textContent = "Reference links refreshed. Press Play again.";
    }).catch((error) => {
        if (speciesId !== selectedSpecies) return;
        status.textContent = `Could not refresh reference media: ${error.message}`;
        retryAction = refreshReferenceMedia;
        retry.hidden = false;
    }).finally(() => { referenceRefresh = null; });
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
        if (cloudMode) {
            try {
                if (!localStorage.getItem("osa-reviewer-id")) {
                    localStorage.setItem("osa-reviewer-id", crypto.randomUUID());
                }
            } catch {
                // Reviewing remains available when storage is disabled.
            }
        }
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
        const navigation = restoredNavigation;
        restoredNavigation = null;
        if (navigation && speciesById.has(navigation.speciesId)) {
            dropdown.value = navigation.speciesId;
            await loadSpecies(navigation.speciesId, navigation.count, navigation.cursor);
        } else await loadSpecies(dropdown.value);
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
            pendingSubmission = createSubmission(batch.clips, changed, reviewerId, reviewerName.value);
        }
        // Persist the exact payload before sending; retries can append duplicates.
        if (!draft) throw new Error("Browser storage is required to safely retain a pending submission.");
        persistDraft();
        updateProgress();
        status.textContent = "Submitting answers...";
        await provider.saveAnnotations(pendingSubmission);
        const labels = { present: "yes", absent: "no", unsure: "unsure" };
        for (const item of pendingSubmission.annotations) submittedAnswers.set(item.candidate_id, labels[item.response]);
        pendingSubmission = null;
        status.textContent = "Answers submitted successfully.";
        try { persistDraft(); }
        catch {
            try { draft.clear(); } catch { /* The server receipt remains authoritative. */ }
            status.textContent += " Browser draft storage could not be updated; avoid retrying this submitted batch.";
        }
    } catch (error) {
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
    try { persistDraft(); } catch { status.textContent = "Could not update the stored draft."; }
    updateProgress();
});
reviewerName.addEventListener("change", () => {
    try { persistDraft(); } catch { /* Submission reports storage failures before sending. */ }
});
$("#logout-button").addEventListener("click", () => provider.api.logout());
$("#login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("#login-button");
    if (button.disabled) return;
    button.disabled = true;
    const password = $("#reviewer-password");
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
        button.disabled = false;
    }
});

loginPanel.hidden = !cloudMode;
workspace.hidden = cloudMode;
$("#logout-button").hidden = !cloudMode;
$("#reviewer-controls").hidden = !cloudMode;
reviewerName.hidden = !cloudMode;
if (!cloudMode) initialize();
