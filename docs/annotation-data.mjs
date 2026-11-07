import { AnnotatorSession } from "./annotator-session.mjs";

export class CandidateCursorError extends Error {
    constructor() {
        super("This review cursor is no longer valid. Restart from the first batch.");
        this.name = "CandidateCursorError";
    }
}

/**
 * Data boundary for the annotation page. Both providers expose the same async
 * methods so the UI can use demo data, a local server, or a remote server.
 * Response shapes and backend requirements are documented in ANNOTATION_API.md.
 */
export class ServerProvider {
    constructor(baseUrl, storage = getSessionStorage(), api = new AnnotatorSession(baseUrl)) {
        this.baseUrl = baseUrl.replace(/\/$/, "");
        this.storage = storage;
        this.api = api;
        this.cacheKey = `annotation-catalog-v1:${this.baseUrl}`;

        const cached = this.readCache();
        this.species = cached?.species;
        this.references = cached?.references;
    }

    readCache() {
        try {
            const cached = JSON.parse(this.storage?.getItem(this.cacheKey) || "null");
            if (!Array.isArray(cached?.species)) return null;
            if (cached.species.some((item) => !item || typeof item.id !== "string" ||
                typeof item.scientificName !== "string" || typeof item.commonName !== "string")) return null;
            if (cached.references && (!Array.isArray(cached.references.entries) ||
                cached.references.entries.some((item) => !item || !Array.isArray(item.clips)))) {
                delete cached.references;
            }

            const remaining = Date.parse(cached.references?.urlsExpireAt) - Date.now();
            if (!(remaining >= 60_000)) delete cached.references;

            return cached;
        } catch {
            return null;
        }
    }

    writeCache() {
        // Session-only storage; expired signed URLs are never reused.
        try {
            this.storage?.setItem(this.cacheKey, JSON.stringify({
                species: this.species,
                references: this.references,
            }));
        } catch {
            // Storage restrictions must not prevent catalog loading.
        }
    }

    async listSpecies() {
        if (!this.candidateSpeciesPromise) {
            // Queue availability can change between visits; do not persist this list.
            this.candidateSpeciesPromise = this.loadSpecies().catch((error) => {
                this.candidateSpeciesPromise = null;
                throw error;
            });
        }
        return this.candidateSpeciesPromise;
    }

    async loadSpecies() {
        const data = await this.getJson("/species?has_audio=true&has_candidates=true");
        if (!Array.isArray(data?.species)) throw new Error("Invalid species response.");
        return data.species.map(parseSpecies);
    }

    async getAnnotationSummary() {
        // Counts must be fresh, independent of reference/catalog caches.
        const data = await this.getJson("/species");
        if (!Array.isArray(data?.species)) throw new Error("Invalid species response.");
        return data.species.map((item) => {
            const species = parseSpecies(item);
            const counts = item.annotation_counts;
            if (!counts || ["positive", "negative", "unsure", "total"].some(
                (key) => !Number.isSafeInteger(counts[key]) || counts[key] < 0,
            ) || counts.total !== counts.positive + counts.negative + counts.unsure) {
                throw new Error("Invalid annotation counts.");
            }
            return { ...species, counts };
        });
    }

    async listReferenceSpecies() {
        if (this.species) return this.species;
        if (this.speciesPromise) return this.speciesPromise;

        // Share concurrent requests; allow retries after failure.
        this.speciesPromise = this.loadReferenceSpecies().catch((error) => {
            this.speciesPromise = null;
            throw error;
        });

        return this.speciesPromise;
    }

    async loadReferenceSpecies() {
        const data = await this.getJson("/species?has_audio=true");
        if (!Array.isArray(data?.species)) {
            throw new Error("Invalid species response.");
        }

        this.species = data.species.map(parseSpecies);
        this.writeCache();

        return this.species;
    }

    async getReferenceSounds(speciesId) {
        const species = await this.listReferenceSpecies();
        const target = species.find((item) => item.id === speciesId);
        if (!target) throw new Error("Unknown target species.");

        const references = await this.getReferences(species);

        return {
            target: references.entries.find((item) => item.id === speciesId),
            comparisons: references.entries.filter((item) => item.id !== speciesId),
        };
    }

    async getReferences(species) {
        if (this.referencesAreFresh()) return this.references;
        if (this.referencesPromise) return this.referencesPromise;

        this.referencesPromise = this.loadReferences(species).finally(() => {
            this.referencesPromise = null;
        });

        return this.referencesPromise;
    }

    referencesAreFresh() {
        if (!this.references) return false;

        const remaining = Date.parse(this.references.urlsExpireAt) - Date.now();
        return remaining >= 60_000;
    }

    async loadReferences(species) {
        const query = new URLSearchParams();
        species.forEach((item) => query.append("species_id", item.id));

        const data = await this.getJson(`/references?${query}`);
        if (!data?.references || !Number.isFinite(Date.parse(data.urls_expire_at))) {
            throw new Error("Invalid reference response.");
        }

        const entries = species.map((item) => {
            const clips = data.references[item.id];
            if (!Array.isArray(clips)) {
                throw new Error("Invalid reference response.");
            }

            return {
                ...item,
                urlsExpireAt: data.urls_expire_at,
                clips: clips.map(parseReferenceClip),
            };
        });

        // Replace the cache only after the entire response passes validation.
        this.references = { entries, urlsExpireAt: data.urls_expire_at };
        this.writeCache();
        return this.references;
    }

    async getJson(path, signal) {
        const response = await this.api.request(path, {
            signal,
            cache: "no-store",
            credentials: "omit",
        });
        if (!response.ok) {
            const error = new Error(`Catalog request failed (${response.status}).`);
            error.status = response.status;
            throw error;
        }

        return response.json();
    }

    async getCandidates(speciesId, { count = 15, cursor, signal, restartIfEmpty = false } = {}) {
        const query = new URLSearchParams({ species_id: speciesId, count: String(count) });
        if (cursor != null) query.set("cursor", cursor);

        let data;
        try {
            data = await this.getJson(`/candidates?${query}`, signal);
        } catch (error) {
            if (cursor != null && error.status === 400) throw new CandidateCursorError();
            throw error;
        }
        if (
            !Array.isArray(data?.candidates) ||
            !(data.next_cursor === null || typeof data.next_cursor === "string") ||
            !Number.isFinite(Date.parse(data.urls_expire_at))
        ) throw new Error("Invalid candidate response.");

        // A saved cursor can outlive the candidate queue it came from.
        if (restartIfEmpty && cursor != null && !data.candidates.length) {
            return this.getCandidates(speciesId, { count, signal });
        }

        const clips = data.candidates.map((item) => parseCandidate(item, speciesId));
        if (new Set(clips.map((clip) => clip.id)).size !== clips.length) {
            throw new Error("Duplicate candidate IDs.");
        }

        return {
            speciesId, count, inputCursor: cursor ?? null,
            nextCursor: data.next_cursor,
            urlsExpireAt: data.urls_expire_at,
            clips,
        };
    }

    async saveAnnotations(submission) {
        const response = await this.api.request("/annotations", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(submission),
        });
        if (!response.ok) {
            const error = new Error(response.status >= 500
                ? "Outcome uncertain: the write may have succeeded. Manual retry may create duplicates."
                : `Submission rejected (${response.status}).`);
            error.status = response.status;
            throw error;
        }
        const receipt = await response.json();
        if (receipt.submission_id !== submission.submission_id || receipt.accepted_count !== submission.annotations.length) {
            throw new Error("Outcome uncertain: the server did not confirm the complete submission.");
        }
        return receipt;
    }

}

function parseCandidate(item, speciesId) {
    if (
        !item || typeof item.candidate_id !== "string" || !item.candidate_id ||
        typeof item.window_id !== "string" || !item.window_id ||
        typeof item.recording_id !== "string" || !item.recording_id ||
        item.query_species_id !== speciesId || typeof item.audio_url !== "string" || !item.audio_url ||
        !(item.spectrogram_url == null || typeof item.spectrogram_url === "string")
    ) throw new Error("Invalid candidate response.");

    validateCandidateBounds(item);

    return {
        id: item.candidate_id,
        windowId: item.window_id,
        querySpeciesId: item.query_species_id,
        recordingId: item.recording_id,
        createdAt: item.created_at,
        score: item.score,
        offsets: [0, 5],
        // Source-recording bounds are metadata, never playback/spectrogram bounds.
        recordingOffsets: [item.recording_start_seconds, item.recording_end_seconds],
        audioUrl: item.audio_url,
        spectrogramUrl: item.spectrogram_url || null,
    };
}

function validateCandidateBounds(item) {
    if (item.start_seconds !== 0 || item.end_seconds !== 5) {
        throw new Error("Invalid candidate response: expected standalone 5-second window audio.");
    }

    const start = item.recording_start_seconds;
    const end = item.recording_end_seconds;
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 ||
        Math.abs(end - start - 5) > 0.000001) {
        throw new Error("Invalid candidate response: missing or invalid source-recording bounds.");
    }
}

function parseSpecies(item) {
    if (
        !item ||
        typeof item.species_id !== "string" ||
        !item.species_id ||
        typeof item.scientific_name !== "string" ||
        typeof item.common_name !== "string"
    ) {
        throw new Error("Invalid species response.");
    }

    return {
        id: item.species_id,
        scientificName: item.scientific_name,
        commonName: item.common_name,
    };
}

function parseReferenceClip(clip) {
    if (
        !clip ||
        typeof clip.reference_id !== "string" ||
        typeof clip.audio_url !== "string" ||
        !(clip.spectrogram_url == null || typeof clip.spectrogram_url === "string")
    ) {
        throw new Error("Invalid reference response.");
    }

    return {
        id: clip.reference_id,
        audioUrl: clip.audio_url,
        spectrogramUrl: clip.spectrogram_url || null,
    };
}

// Each sample is [window ID, source recording ID, start seconds, end seconds, source URI].
// Demo mode reuses the same WAV/PNG pairs for every target species.
const windows = [
    [25435, 633, 35, 40, "sw2/1Jsw2_20261009_014000.wav"],
    [23933, 508, 25, 30, "sw1/1Jsw1_20260919_224000.wav"],
    [23976, 512, 0, 5, "sw1/1Jsw1_20260920_000000.wav"],
    [24001, 514, 5, 10, "sw1/1Jsw1_20260920_004000.wav"],
    [22970, 428, 10, 15, "sw1/1Jsw1_20260917_225400.wav"],
];
export class DemoProvider {
    async listSpecies() {
        // Share one CSV request across species and reference loading. A failed
        // request clears the cached promise below so the next attempt can retry.
        if (!this.speciesPromise)
            this.speciesPromise = new Promise((resolve, reject) => {
                if (!globalThis.Papa) {
                    reject(
                        new Error(
                            "Species parser unavailable. Check your connection and retry.",
                        ),
                    );
                    return;
                }
                Papa.parse("../specieslist.csv", {
                    download: true,
                    header: true,
                    delimiter: "\t",
                    skipEmptyLines: true,
                    error: reject,
                    complete: ({ data, errors }) => {
                        if (errors.length) {
                            reject(
                                new Error("Species data could not be read."),
                            );
                            return;
                        }
                        resolve(
                            data
                                .filter((row) => row.display === "1")
                                .map((row) => {
                                    const root = `static/${encodeURIComponent(row.scientific_name)}`;
                                    // Media columns contain comma-separated lists inside the TSV table.
                                    const files = row.audio_files
                                        ? Papa.parse(row.audio_files).data[0]
                                        : [];
                                    const specs = row.spec_files
                                        ? Papa.parse(row.spec_files).data[0]
                                        : [];
                                    return {
                                        id: row.scientific_name,
                                        scientificName: row.scientific_name,
                                        commonName: row.common_name,
                                        imageUrl: `${root}/img/${encodeURIComponent(row.scientific_name)}0.webp`,
                                        clips: files
                                            .filter(Boolean)
                                            .map((file, i) => ({
                                                audioUrl: `${root}/audio/${encodeURIComponent(file)}`,
                                                spectrogramUrl: specs[i]
                                                    ? `${root}/img/spec/${encodeURIComponent(specs[i])}`
                                                    : null,
                                            })),
                                    };
                                }),
                        );
                    },
                });
            }).catch((error) => {
                this.speciesPromise = null;
                throw error;
            });
        return this.speciesPromise;
    }

    async getReferenceSounds(speciesId) {
        const species = await this.listSpecies();
        const target = species.find((item) => item.id === speciesId);
        if (!target) throw new Error("Unknown target species.");
        return {
            target,
            comparisons: species.filter((item) => item.id !== speciesId),
        };
    }

    async getCandidates(speciesId) {
        return {
            batchId: `demo:${speciesId}`,
            clips: windows.map(
                ([windowId, recordingId, start, end, uri], i) => ({
                    id: String(windowId),
                    recordingId,
                    offsets: [start, end],
                    uri,
                    datasetName: "grillos",
                    audioUrl: `fake_data/download${i}.wav`,
                    spectrogramUrl: `fake_data/output${i}.png`,
                }),
            ),
        };
    }

    async saveAnnotations() {
        throw new Error("Saving is unavailable in demo mode. Answers stay in this page only.");
    }

}

export function createProvider(config) {
    if (config.provider === "demo") return new DemoProvider();
    if (config.provider === "server") return new ServerProvider(config.baseUrl, null);
    throw new Error(`Unknown annotation provider: ${config.provider}`);
}

// Keep source metadata and omit unanswered windows. The UI passes only changed
// answers and reuses submissionId when retrying an unchanged failed submission.
export function buildSubmission(
    speciesId,
    batchId,
    clips,
    answers,
    submissionId,
) {
    return {
        submissionId,
        speciesId,
        batchId,
        createdAt: new Date().toISOString(),
        annotations: clips
            .filter((item) => answers.has(item.id))
            .map((item) => ({
                windowId: item.id,
                recordingId: item.recordingId,
                offsets: item.offsets,
                uri: item.uri,
                datasetName: item.datasetName,
                label: answers.get(item.id),
            })),
    };
}

function getSessionStorage() {
    try {
        return globalThis.sessionStorage;
    } catch {
        return null;
    }
}
