# Annotation server integration

The existing annotation page uses four async methods in `annotation-data.mjs`.
`annotation-config.js` selects `server` at `http://127.0.0.1:8080/api` by default.
Select `demo` to use the CSV/static sample provider. No authentication headers,
Google credentials, browser database access, or new runtime dependencies are needed.
The backend assigns the local annotator (`local-development`); this is attribution,
not authentication. Never silently fall back to demo after a server error.

## Run locally

Check `http://127.0.0.1:8080/health` before starting another backend. Its response
is `{"status":"ok","storage":"sqlite","schemaVersion":5}`. From `C:/agile`:

```powershell
.venv/Scripts/python.exe -m uvicorn server.app.main:create_app --factory --host 127.0.0.1 --port 8080
```

From `C:/OsaAnurans`, run `python -m http.server 8000 --bind 127.0.0.1` and open
`http://127.0.0.1:8000/docs/annotate.html`, not a file URL. Backend CORS allows
localhost and 127.0.0.1 on port 8000. DENEBR is selected initially when available;
the dropdown displays scientific names with canonical codes as option values.

## Implemented provider contract

Paths are relative to the configured base URL; trailing slashes are removed.
Identifiers remain opaque strings and are encoded with `encodeURIComponent`.
The adapter returns validated responses without remapping or dropping metadata.

- `listSpecies()` calls `GET /species`: an array of `{id, scientificName, commonName}`.
  The full catalog has 52 species. Use its names and IDs; do not maintain another mapping.
- `getReferenceSounds(id)` calls `GET /species/{id}/references`:
  `{target, comparisons}`. Each species contains `clips`, each with `audioUrl`,
  nullable `spectrogramUrl`, and nullable playback bounds. `imageUrl` is optional.
  Comparisons include species with empty clip lists. DENEBR has three target examples.
  References play their whole files with native audio controls.
- `getCandidates(id)` calls `GET /species/{id}/candidates`:
  `{batchId, clips}` with at most 15 unique windows. Each clip has string `id`,
  numeric `recordingId`, `offsets: [start, end]`, `uri`, `datasetName`, `audioUrl`,
  nullable `spectrogramUrl`, finite increasing `startSeconds`/`endSeconds` matching
  the offsets, and nonnegative integer `numAnnotations`. The latter counts rows,
  including revisions, rather than distinct reviewers. No work is
  `{batchId: null, clips: []}` and is a normal state.
- `saveAnnotations(submission)` calls `POST /annotations` with JSON and
  `Content-Type: application/json`. It requires an exact
  `{submissionId, savedCount}` acknowledgement matching the submitted snapshot.
  Partial acknowledgements, malformed JSON, timeouts, and HTTP errors reject.

```json
{
  "submissionId": "client-generated-UUID",
  "speciesId": "DENEBR",
  "batchId": "opaque-server-issued-context",
  "createdAt": "2026-11-02T12:00:00.000Z",
  "annotations": [{
    "windowId": "osa-local:window:24898",
    "recordingId": 588,
    "offsets": [50, 55],
    "uri": "sw2/1Jsw2_20261008_000000.wav",
    "datasetName": "grillos",
    "label": "yes"
  }]
}
```

`buildSubmission` preserves source fields and includes only changed answered
windows, with labels exactly `yes`, `no`, or `unsure`. Send 1-15 answers with
unique window IDs, a UTC timestamp, and no annotator identity or extra fields.
The backend uses its own timestamp. Saves are atomic. Exact retries under the
same submission ID are idempotent, including after restart; changing answers,
metadata, species, or context under that ID produces 409. New IDs append audit
rows; the latest annotation per person/window/species is current.

The shared request helper supports injected fetch and a finite 15-second default
AbortController timeout, including response parsing; timers are cleared on completion.
API errors use `{"error":{"message":"..."}}`, sometimes with validation details.
404, 409, 422, 503 and non-JSON errors are surfaced without marking answers saved.

## Save, retry, and advancement

Saving locks answers and species selection. A failure preserves answers and the
unchanged submission snapshot, including UUID and timestamp. Retrying sends the
same payload. Editing an answer creates a new submission; revisions append audit
history. Species changes and leaving the page warn about unsaved answers.

After an acknowledged save, `Load remaining work` fetches candidates again.
It is disabled while loading, saving, or holding unsaved answers or a pending
submission. There is no cursor: batch IDs identify species/annotator context,
not frozen assignments. Every saved label, including unsure, removes its window
from this annotator's queue. Unanswered windows remain available. Outgoing audio
is paused before candidate replacement. A refresh failure keeps the current cards
and explicitly reports that the save succeeded, with a separate load retry.

Optional recovery `GET /submissions/{submissionId}` returns the original ack or
404 for this annotator. The basic client retries the unchanged POST. A recovery
404 does not prove an earlier timed-out POST cannot still commit.

## Media

Always use returned asset URLs. Candidate audio serves the entire source WAV
with HTTP byte-range support. `audio-window.mjs` seeks to the review start, stops
at the end, and replays; it does not download/decode/trim a complete recording.
Bounds are approximate playback controls, not access restrictions. Reference and
candidate playback are mutually exclusive. Missing spectrograms/images are omitted.

## Verification

```powershell
node --test tests/annotation-data.test.mjs tests/server-provider.test.mjs tests/annotation-save.test.mjs tests/audio-window.test.mjs
```

The existing DemoProvider has user edits that log saved batches and clear storage.
Its persistence test currently fails (`this.storage.clear is not a function` in
the test storage). These unrelated edits are preserved; demo persistence needs a
separate fix. Demo mode still uses the existing PapaParse CDN dependency.

`tests/annotation-browser.mjs` runs optional real Chrome acceptance through CDP
on port 9222, frontend on 8000, and an isolated fixture backend on 8081. It injects
the test base URL into the browser without changing production configuration.
It exercises references, playback, all labels, saves, queue refresh, simulated
network failures and invalid acknowledgements, exact retries, and empty species.
It writes `.annotation-test-local/browser-empty-state.png`. Use a fresh test store
for each run because successful saves consume its queue:

```powershell
# From C:/agile; choose a fresh directory inside the frontend workspace.
$env:OSA_SERVER_LOCAL = 'C:/OsaAnurans/.annotation-test-local'
.venv/Scripts/python.exe -m server.tools.import_candidates server/local/bundles/denebr-recordings/manifest.json
.venv/Scripts/python.exe -m server.tools.import_references C:/OsaAnurans
.venv/Scripts/python.exe -m uvicorn server.app.main:create_app --factory --host 127.0.0.1 --port 8081
```

Start Chrome headless with a separate profile, remote debugging port 9222 and
`--autoplay-policy=no-user-gesture-required`, then run `node tests/annotation-browser.mjs`.
Never delete the normal server database to reset tests. Test labels stay in the
isolated application store; the original Hoplite research database is unchanged.
