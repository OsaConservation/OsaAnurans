/**
 * Data boundary for the annotation page. Both providers expose the same async
 * methods so the UI can use demo data, a local server, or a remote server.
 * Response shapes and backend requirements are documented in ANNOTATION_API.md.
 */
export class ServerProvider {
    constructor(baseUrl) {
        this.baseUrl = baseUrl.replace(/\/$/, "");
    }

    async listSpecies() {
        // TODO: GET /species; authenticate and validate response.
        throw new Error("Server species loading is not implemented yet.");
    }

    async getReferenceSounds(speciesId) {
        // TODO: GET /species/:speciesId/references; URL-encode speciesId.
        throw new Error("Server reference loading is not implemented yet.");
    }

    async getRecordings(speciesId) {
        // TODO: GET /species/:speciesId/recordings; return a stable batch ID.
        throw new Error("Server recording loading is not implemented yet.");
    }

    async saveAnnotations(batch) {
        // TODO: POST /annotations using submissionId for idempotent retries.
        // Resolve only after the entire batch is durably saved. Reject on errors.
        throw new Error("Server annotation saving is not implemented yet.");
    }
}

// Each sample is [window ID, recording ID, start seconds, end seconds, source URI].
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
                                        examples: files
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

    async getRecordings(speciesId) {
        return {
            batchId: `demo:${speciesId}`,
            recordings: windows.map(
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
    recordings,
    answers,
    submissionId,
) {
    return {
        submissionId,
        speciesId,
        batchId,
        createdAt: new Date().toISOString(),
        annotations: recordings
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
