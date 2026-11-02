/**
 * Data boundary for the annotation page. Both providers expose the same async
 * methods so the UI can use demo data, a local server, or a remote server.
 * Response shapes and backend requirements are documented in ANNOTATION_API.md.
 */
export class ServerProvider {
    constructor(
        baseUrl,
        { fetch: fetchImpl = globalThis.fetch, timeoutMs = 15000 } = {},
    ) {
        if (typeof baseUrl !== "string" || !baseUrl.trim())
            throw new Error("Server base URL is required.");
        if (!Number.isFinite(timeoutMs) || timeoutMs <= 0)
            throw new Error("Invalid request timeout.");

        this.baseUrl = baseUrl.replace(/\/+$/, "");
        // Browser fetch must be called without binding it to this provider.
        this.fetch = (...args) => fetchImpl(...args);
        this.timeoutMs = timeoutMs;
    }

    // Apply one deadline to fetching headers and reading the response body.
    async request(path, body) {
        const controller = new AbortController();
        let timer;
        const timeout = new Promise((_, reject) => {
            timer = setTimeout(() => {
                controller.abort();
                reject(
                    new Error(
                        "Server request timed out. Retry the unchanged submission if saving.",
                    ),
                );
            }, this.timeoutMs);
        });

        try {
            return await Promise.race([
                timeout,
                this.fetchJson(path, body, controller.signal),
            ]);
        } catch (error) {
            throw new Error(`Annotation server: ${error.message}`, {
                cause: error,
            });
        } finally {
            clearTimeout(timer);
        }
    }

    async fetchJson(path, body, signal) {
        const options = { method: "GET", signal };
        if (body !== undefined) {
            options.method = "POST";
            options.headers = { "Content-Type": "application/json" };
            options.body = JSON.stringify(body);
        }

        const response = await this.fetch(`${this.baseUrl}${path}`, options);
        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error(
                response.ok
                    ? "Server returned invalid JSON."
                    : `Server HTTP ${response.status}: non-JSON response.`,
            );
        }

        if (!response.ok) {
            const message = data?.error?.message || "Request failed.";
            throw new Error(`Server HTTP ${response.status}: ${message}`);
        }
        return data;
    }

    async listSpecies() {
        const data = await this.request("/species");
        requireShape(Array.isArray(data), "species list");
        data.forEach(validateSpecies);
        validateUniqueIds(data);
        return data;
    }

    async getReferenceSounds(speciesId) {
        const data = await this.request(
            `/species/${encodeURIComponent(speciesId)}/references`,
        );
        requireShape(data && Array.isArray(data.comparisons), "references");
        requireShape(data.target?.id === speciesId, "reference target");
        const species = [data.target, ...data.comparisons];
        validateUniqueIds(species);
        species.forEach(validateReferenceSpecies);
        return data;
    }

    async getCandidates(speciesId) {
        const data = await this.request(
            `/species/${encodeURIComponent(speciesId)}/candidates`,
        );
        requireShape(
            data && Array.isArray(data.clips) && data.clips.length <= 15,
            "candidate clips",
        );
        requireShape(
            isNonemptyString(data.batchId) ||
                (data.batchId === null && !data.clips.length),
            "batch ID",
        );
        validateUniqueIds(data.clips);
        data.clips.forEach(validateCandidate);
        return data;
    }

    async saveAnnotations(batch) {
        const data = await this.request("/annotations", batch);
        requireShape(
            data &&
                data.submissionId === batch.submissionId &&
                Number.isInteger(data.savedCount) &&
                data.savedCount === batch.annotations.length,
            "complete save acknowledgement",
        );
        return data;
    }
}

// Validate at the provider boundary so rendering receives predictable data.
function requireShape(condition, field) {
    if (!condition) throw new Error(`Invalid server response: ${field}.`);
}

const isNonemptyString = (value) =>
    typeof value === "string" && value.trim().length > 0;

function validateOptionalMedia(value) {
    requireShape(
        value == null || isNonemptyString(value),
        "optional media URL",
    );
}

function validateSpecies(item) {
    requireShape(
        item &&
            isNonemptyString(item.id) &&
            isNonemptyString(item.scientificName) &&
            typeof item.commonName === "string",
        "species",
    );
}

function validateUniqueIds(items) {
    requireShape(
        items.every((item) => item && isNonemptyString(item.id)) &&
            new Set(items.map((item) => item.id)).size === items.length,
        "unique IDs",
    );
}

function validateBounds(start, end) {
    requireShape(
        Number.isFinite(start) &&
            Number.isFinite(end) &&
            start >= 0 &&
            end > start,
        "playback bounds",
    );
}

function validateReferenceSpecies(species) {
    validateSpecies(species);
    validateOptionalMedia(species.imageUrl);
    requireShape(Array.isArray(species.clips), "reference clips");

    species.clips.forEach((clip) => {
        requireShape(
            clip && isNonemptyString(clip.audioUrl),
            "reference audio URL",
        );
        validateOptionalMedia(clip.spectrogramUrl);

        // Whole-file reference clips normally have no playback bounds.
        if (clip.startSeconds != null || clip.endSeconds != null) {
            validateBounds(clip.startSeconds, clip.endSeconds);
        }
    });
}

function validateCandidate(clip) {
    requireShape(
        Number.isFinite(clip.recordingId) && Number.isInteger(clip.recordingId),
        "recording ID",
    );
    requireShape(
        isNonemptyString(clip.uri) &&
            isNonemptyString(clip.datasetName) &&
            isNonemptyString(clip.audioUrl),
        "candidate source fields",
    );
    validateOptionalMedia(clip.spectrogramUrl);

    // Playback bounds and saved source offsets describe the same review window.
    requireShape(
        Array.isArray(clip.offsets) && clip.offsets.length === 2,
        "offsets",
    );
    validateBounds(clip.startSeconds, clip.endSeconds);
    requireShape(
        clip.offsets[0] === clip.startSeconds &&
            clip.offsets[1] === clip.endSeconds,
        "matching offsets",
    );
    requireShape(
        Number.isInteger(clip.numAnnotations) && clip.numAnnotations >= 0,
        "annotation count",
    );
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
    // Injectable storage lets tests simulate persistence errors without a browser.
    constructor(storage = window.localStorage) {
        this.storage = storage;
    }

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

    async saveAnnotations(batch) {
        // Browser-only demo storage. Repeated submission IDs do not add duplicate batches.
        const key = "osa-demo-annotations-v1";
        const saved = JSON.parse(this.storage.getItem(key) || "[]");
        if (!saved.some((item) => item.submissionId === batch.submissionId)) {
            saved.push(batch);
            this.storage.setItem(key, JSON.stringify(saved));
        }
        console.log("Saved");
        console.log(saved);
        this.storage.clear();

        return {
            submissionId: batch.submissionId,
            savedCount: batch.annotations.length,
        };
    }
}

export function createProvider(config) {
    if (config.provider === "demo") return new DemoProvider();
    if (config.provider === "server") return new ServerProvider(config.baseUrl);
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
