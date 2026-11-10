const PAGE_SIZE = 50;
const missing = "Unavailable";

export function annotatorLabel(row) {
    return row.annotator_name?.trim() || missing;
}

export function filterRows(rows, { species = "", response = "", annotator = "", search = "" } = {}) {
    const query = search.trim().toLowerCase();
    return rows.filter((row) =>
        (!species || row.species_id === species) &&
        (!response || row.response === response) &&
        (!annotator || annotatorLabel(row) === annotator) &&
        (!query || `${row.recording_filename ?? ""} ${row.window_id ?? ""}`.toLowerCase().includes(query)));
}

export function sortRows(rows) {
    const timestamp = (row) => Date.parse(row.created_at) || 0;
    return [...rows].sort((a, b) => timestamp(b) - timestamp(a) || String(a.row_id).localeCompare(String(b.row_id), undefined, { numeric: true }));
}

export function pageRows(rows, page, size = PAGE_SIZE) {
    return rows.slice((page - 1) * size, page * size);
}

export function resultCounts(rows) {
    return { annotations: rows.length, clips: new Set(rows.map((row) => row.window_id).filter(Boolean)).size };
}

function formatDate(value) {
    if (!value || !Number.isFinite(Date.parse(value))) return missing;
    return new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function sourceOffset(row) {
    if (row.start_seconds == null) return "Source offset unavailable";
    const end = row.end_seconds == null ? "" : `–${Number(row.end_seconds).toFixed(1)}`;
    return `${Number(row.start_seconds).toFixed(1)}${end} s in recording`;
}

export async function startBrowser() {
    const byId = (id) => document.getElementById(id);
    const elements = Object.fromEntries([
        "species-filter", "response-filter", "annotator-filter", "search", "annotation-rows",
        "result-count", "previous-page", "next-page", "page-status", "clip-audio",
        "spectrogram", "media-status", "selected-title", "selected-summary", "annotation-details",
        "snapshot-status", "load-status",
    ].map((id) => [id, byId(id)]));
    let rows = [];
    let filtered = [];
    let page = 1;
    let selected = null;
    let generation = 0;
    let snapshotMessage = "";
    let sharedSpectrogram;
    const audio = elements["clip-audio"];

    function clearSelection() {
        generation++;
        selected = null;
        audio.pause();
        audio.removeAttribute("src");
        audio.load();
        audio.hidden = true;
        elements.spectrogram.replaceChildren();
        elements["media-status"].textContent = "";
        elements["selected-title"].textContent = "Choose an annotation";
        elements["selected-summary"].textContent = "Select a species in the table to listen to its clip.";
        elements["annotation-details"].replaceChildren();
    }

    function addDetail(label, value) {
        const term = document.createElement("dt");
        term.textContent = label;
        const detail = document.createElement("dd");
        detail.textContent = value || missing;
        elements["annotation-details"].append(term, detail);
    }

    async function selectRow(row) {
        const sameMedia = selected?.audio_url && selected.audio_url === row.audio_url;
        if (!sameMedia) clearSelection();
        selected = row;
        for (const tr of elements["annotation-rows"].children) {
            const active = tr.dataset.rowId === String(row.row_id);
            tr.setAttribute("aria-selected", String(active));
            tr.querySelector("button").setAttribute("aria-pressed", String(active));
        }
        elements["selected-title"].textContent = `${row.species_id} · ${row.response}`;
        elements["selected-summary"].textContent = `${annotatorLabel(row)} · ${formatDate(row.created_at)}`;
        elements["annotation-details"].replaceChildren();
        addDetail("Recording", row.recording_filename);
        addDetail("Source position", sourceOffset(row));
        addDetail("Annotation ID", row.annotation_id);
        addDetail("Window ID", row.window_id);
        addDetail("Annotator ID", row.annotator_id);
        if (sameMedia) return;
        if (!row.audio_url) {
            elements["media-status"].textContent = "Local audio is unavailable for this clip. The annotation is still shown.";
            return;
        }
        const current = generation;
        audio.src = row.audio_url;
        audio.hidden = false;
        const generate = async () => {
            if (current !== generation) return;
            elements["media-status"].textContent = "Generating spectrogram…";
            elements.spectrogram.replaceChildren();
            try {
                sharedSpectrogram ??= import("/shared/local-spectrogram.mjs");
                const { generateSpectrogram } = await sharedSpectrogram;
                const url = await generateSpectrogram(
                    { audioUrl: row.audio_url, recordingId: row.window_id },
                    (message) => { if (current === generation) elements["media-status"].textContent = message; },
                    async () => current === generation,
                );
                if (current !== generation) return;
                const image = document.createElement("img");
                image.alt = `Spectrogram of clip ${row.window_id}, 0–12 kHz`;
                image.src = url;
                await image.decode();
                if (current !== generation) return;
                elements.spectrogram.replaceChildren(image);
                elements["media-status"].textContent = "Clip playback starts at 0 s. Dates use your browser’s local timezone.";
            } catch {
                if (current !== generation) return;
                elements["media-status"].textContent = "Could not generate spectrogram. You can still try audio playback.";
                const retry = document.createElement("button");
                retry.type = "button";
                retry.textContent = "Retry spectrogram";
                retry.addEventListener("click", generate);
                elements.spectrogram.replaceChildren(retry);
            }
        };
        await generate();
    }

    function renderRows() {
        const fragment = document.createDocumentFragment();
        for (const row of pageRows(filtered, page)) {
            const tr = document.createElement("tr");
            tr.dataset.rowId = row.row_id;
            tr.setAttribute("aria-selected", String(selected?.row_id === row.row_id));
            const species = document.createElement("td");
            const button = document.createElement("button");
            button.type = "button";
            button.className = "select-annotation";
            button.textContent = row.species_id || missing;
            button.setAttribute("aria-pressed", String(selected?.row_id === row.row_id));
            button.setAttribute("aria-label", `Listen to ${row.species_id}, ${row.response}, ${annotatorLabel(row)}`);
            button.addEventListener("click", () => selectRow(row));
            species.append(button);
            tr.append(species);
            const response = document.createElement("td");
            const badge = document.createElement("span");
            badge.className = `response ${row.response}`;
            badge.textContent = row.response || missing;
            response.append(badge);
            tr.append(response);
            for (const value of [annotatorLabel(row), formatDate(row.created_at)]) {
                const cell = document.createElement("td");
                cell.textContent = value;
                tr.append(cell);
            }
            const recording = document.createElement("td");
            const filename = document.createElement("span");
            filename.className = "recording";
            filename.textContent = row.recording_filename || row.window_id || missing;
            const offset = document.createElement("span");
            offset.className = "muted";
            offset.textContent = sourceOffset(row);
            recording.append(filename, offset);
            tr.append(recording);
            fragment.append(tr);
        }
        elements["annotation-rows"].replaceChildren(fragment);
        const counts = resultCounts(filtered);
        elements["result-count"].textContent = `${counts.annotations} annotations · ${counts.clips} clips`;
        const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
        elements["page-status"].textContent = `Page ${page} of ${pages}`;
        elements["previous-page"].disabled = page === 1;
        elements["next-page"].disabled = page >= pages;
        elements["load-status"].textContent = filtered.length ? "" : rows.length ? "No annotations match these filters." : snapshotMessage || "This snapshot contains no saved annotation rows.";
    }

    function applyFilters() {
        page = 1;
        clearSelection();
        filtered = filterRows(rows, {
            species: elements["species-filter"].value,
            response: elements["response-filter"].value,
            annotator: elements["annotator-filter"].value,
            search: elements.search.value,
        });
        renderRows();
    }

    for (const id of ["species-filter", "response-filter", "annotator-filter"]) {
        elements[id].addEventListener("change", applyFilters);
    }
    elements.search.addEventListener("input", applyFilters);
    for (const [id, direction] of [["previous-page", -1], ["next-page", 1]]) {
        elements[id].addEventListener("click", () => {
            clearSelection();
            page += direction;
            renderRows();
        });
    }
    audio.addEventListener("error", () => {
        if (selected?.audio_url) elements["media-status"].textContent = "Audio playback failed. The local file may be unavailable or unreadable.";
    });

    try {
        const response = await fetch("/api/data");
        if (!response.ok) throw new Error(`Could not load annotations (${response.status}).`);
        const data = await response.json();
        snapshotMessage = data.message || "";
        rows = sortRows(data.annotations);
        elements["snapshot-status"].textContent = `Saved ${formatDate(data.snapshot.created_at)} · ${data.snapshot.source || "Saved snapshot"} · Dates use your browser’s local timezone.`;
        elements["snapshot-status"].title = data.snapshot.path || "";
        for (const [id, values] of [
            ["species-filter", rows.map((row) => row.species_id).filter(Boolean)],
            ["annotator-filter", rows.map(annotatorLabel)],
        ]) {
            for (const value of [...new Set(values)].sort((a, b) => a.localeCompare(b))) {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = value;
                elements[id].append(option);
            }
        }
        applyFilters();
    } catch (error) {
        elements["load-status"].textContent = `${error.message} Restart browse after checking the saved snapshot.`;
        elements["snapshot-status"].textContent = "Snapshot unavailable";
    }
}

if (typeof document !== "undefined") startBrowser();
