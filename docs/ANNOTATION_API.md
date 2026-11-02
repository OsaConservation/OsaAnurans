# Annotation data interface

The page depends on four async provider methods in `annotation-data.mjs`. Configure the provider in `annotation-config.js`. `demo` is functional; `server` intentionally rejects with clear messages until its TODOs are implemented. A local or remote server uses the same adapter and a different `baseUrl`. The browser never connects directly to a database.

## Proposed endpoints and response shapes

All paths below are relative to `baseUrl`. IDs are opaque strings. Media URLs must be browser-accessible, absolute or relative to the annotation page. Server responses must be validated by the future adapter before reaching the UI.

- `listSpecies()` → `GET /species`: `[{id, scientificName, commonName}]`.
- `getReferenceSounds(speciesId)` → `GET /species/:id/references`: `{target: ReferenceSpecies, comparisons: ReferenceSpecies[]}`. Exclude the target from comparisons.
- `ReferenceSpecies`: `{id, scientificName, commonName, imageUrl?: string, examples: [{audioUrl, spectrogramUrl?: string}]}`.
- `getRecordings(speciesId)` → `GET /species/:id/recordings`: `{batchId, recordings: [{id, recordingId, offsets: [startSeconds, endSeconds], uri, datasetName, audioUrl, spectrogramUrl?: string}]}`. Recording IDs must be unique in a batch. An empty array means no work available. The target and batch must stay associated on the server.
- `saveAnnotations(batch)` → `POST /annotations`: request shown below; response `{submissionId, savedCount}`. Resolve only after every annotation is durably saved. Reject HTTP errors, invalid responses, timeouts, and partial saves. The UI keeps answers on failure.

```json
{
  "submissionId": "client-generated UUID",
  "speciesId": "Agalychnis callidryas",
  "batchId": "server-issued batch ID",
  "createdAt": "2026-11-02T12:00:00.000Z",
  "annotations": [{
    "windowId": "25435",
    "recordingId": 633,
    "offsets": [35, 40],
    "uri": "sw2/1Jsw2_20261009_014000.wav",
    "datasetName": "grillos",
    "label": "yes"
  }]
}
```

Labels are `yes` (target audible), `no` (target not audible), or `unsure`. Unanswered recordings are omitted. A save contains only new or changed answers. Subsequent saves can revise earlier labels.

## Server TODOs

Implement fetch/response validation in `ServerProvider`, authentication, timeouts, and CORS for a separate origin. URL-encode species IDs. Derive annotator identity from the authenticated session, and validate window IDs, batch membership, offsets, and labels against trusted server records. Use server timestamps as authoritative.

Make each submission atomic and idempotent by submissionId, returning the original acknowledgement for retries. Upsert revisions per annotator/species/window so changing an answer updates its current value rather than creating contradictory training labels. An audit history can retain revisions. These are proposed semantics, not an implemented backend.

The UI reuses a submissionId for retries of an unchanged payload. Editing after an error creates a new submission; upsert semantics are therefore required even with idempotency. Saves lock the species and answers until completion. Failed requests keep answers editable and display a retry message. Species changes and leaving the page warn when there are unsaved answers.

## Demo behavior

Species and reference media come from the existing CSV/static files. The same five sample recordings are reused for every target, clearly marked as demo data. Successful batches are stored under `osa-demo-annotations-v1` in localStorage; nothing reaches a training database. Storage errors leave answers unsaved. Saved batches remain in storage for inspection, but previous answers are not restored into a new session. There is no automatic batch paging yet; add a cursor/next-batch operation when the server supports it.

Serve from the repository root with `.venv/Scripts/python.exe -m http.server 8000 --bind 127.0.0.1`, then open `/docs/annotate.html`. The existing PapaParse CDN dependency requires a network connection. No new runtime libraries are used.
