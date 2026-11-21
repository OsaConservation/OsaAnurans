// One active species queue. API pages contain metadata only; media loads when rendered.
export class CandidateQueue {
    constructor(provider, speciesId, seed) {
        this.provider = provider;
        this.speciesId = speciesId;
        this.seed = seed;
        this.buffer = [];
        this.cursor = null;
        this.started = false;
    }

    async next(count, signal) {
        if (!Number.isInteger(count) || count < 1 || count > 20) {
            throw new Error("Visible candidate count must be between 1 and 20.");
        }
        // Work on copies so failed/aborted reads cannot consume buffered questions.
        const buffer = [...this.buffer];
        let cursor = this.cursor;
        let started = this.started;
        while (buffer.length < count && (!started || cursor !== null)) {
            const page = await this.provider.getCandidates(this.speciesId, { count: 20, cursor, signal, seed: this.seed });
            buffer.push(...page.clips.map((clip) => ({ clip, expiresAt: page.urlsExpireAt })));
            if (page.nextCursor !== null && page.nextCursor === cursor) {
                throw new Error("Candidate cursor did not advance. Restart this species.");
            }
            cursor = page.nextCursor;
            started = true;
        }
        if (signal?.aborted) throw new DOMException("Request aborted", "AbortError");
        const visible = buffer.splice(0, count);
        if (new Set(visible.map(({ clip }) => clip.id)).size !== visible.length) {
            throw new Error("Duplicate candidate IDs. Restart this species.");
        }
        let batch = {
            speciesId: this.speciesId, count, clips: visible.map(({ clip }) => clip),
            hasNext: buffer.length > 0 || cursor !== null,
            urlsExpireAt: visible.length
                ? new Date(Math.min(...visible.map(({ expiresAt }) => Date.parse(expiresAt)))).toISOString()
                : new Date().toISOString(),
        };
        if (visible.length && Date.parse(batch.urlsExpireAt) - Date.now() < 60_000) {
            batch = await this.provider.refreshCandidateMedia(batch, signal);
        }
        if (signal?.aborted) throw new DOMException("Request aborted", "AbortError");
        this.buffer = buffer;
        this.cursor = cursor;
        this.started = started;
        return batch;
    }
}
