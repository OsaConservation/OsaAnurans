const responses = { yes: "present", no: "absent", unsure: "unsure" };

export function createSubmission(clips, answers, reviewerId, name, id = crypto.randomUUID()) {
    const annotations = clips.filter((clip) => answers.has(clip.id)).map((clip) => ({
        candidate_id: clip.id,
        window_id: clip.windowId,
        query_species_id: clip.querySpeciesId,
        response: responses[answers.get(clip.id)],
    }));
    if (!name.trim() || name.trim().length > 100) throw new Error("Enter your name (up to 100 characters).");
    if (!annotations.length || annotations.length > 20) throw new Error("Answer between 1 and 20 clips.");
    return { submission_id: id, annotator_id: reviewerId, annotator_name: name.trim(), annotations };
}

// Only IDs, answers, and navigation are persisted; never media URLs or tokens.
export class SubmissionDraft {
    constructor(storage, key) {
        this.storage = storage;
        this.key = key;
    }
    read() {
        try { return JSON.parse(this.storage.getItem(this.key) || "null"); }
        catch { return null; }
    }
    write(value) {
        this.storage.setItem(this.key, JSON.stringify(value));
    }
    clear() { this.storage.removeItem(this.key); }
}
