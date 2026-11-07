# Annotation frontend data boundary

`annotation-config.js` selects demo or server mode. The server base URL points
to the deployed Cloud Run annotation service. The page is static on GitHub Pages;
Cloud Run enforces annotator access. The page has `noindex, nofollow` metadata.

## Annotator access

`annotator-session.mjs` posts the entered password unchanged to `/auth/login`.
The password is cleared after each attempt. Four-hour bearer tokens remain in
memory and are attached only to the known Cloud Run API routes. Tokens never go
to GCS, localStorage, URLs, or drafts. Refreshing the page requires login again.

A 401 clears the session and shows the gate without discarding answers, annotator
identity/name, pagination, or a pending submission. Logout hides the workspace,
pauses audio, and clears the token. Reauthentication resumes reads but never
replays a submission automatically. Already-issued signed media URLs retain
their own expiration after logout.

## Catalog and candidates

`ServerProvider` maps API snake_case fields into the frontend model:

- `/species?has_audio=true&has_candidates=true` supplies the target selector.
  It requires reference audio and a candidate with window audio. This list is
  fetched per page visit because publishing can change the queue.
- `/species?has_audio=true` supplies the cached list for comparison references,
  including species without candidates.
- `/references` uses repeated `species_id` parameters. All other species with
  audio appear as comparisons. Reference URLs refresh near expiration.
- `/candidates` receives species, count (slider defaults to 5), and an optional
  opaque cursor. Candidates are ordered by ascending `(score, candidate_id)`;
  smaller scores receive review priority. The adapter retains input cursor,
  next cursor, and expiration without interpreting the cursor.

Candidate IDs key answers; window and query-species IDs are retained separately.
A null next cursor disables Next. Failed requests retain the current batch and
retry the failed request. Obsolete responses cannot replace newer selections.

Candidate `audio_url` values point to standalone 5-second window WAVs, signed
from `windows.audio_gcs_uri`. API `start_seconds` and `end_seconds` must be 0 and
5. Original recording bounds are required separately as
`recording_start_seconds` and `recording_end_seconds` (a 5-second interval).
The adapter exposes playback bounds as `offsets: [0, 5]` and source metadata as
`recordingOffsets`. It rejects responses using the old full-recording contract.
`candidate-player.mjs` and `audio-window.mjs` play the window starting at zero;
recording offsets must never be applied a second time.

Missing spectrogram URLs trigger local generation from that same `[0, 5]` audio
interval as cards
become visible. Analysis, canvas rendering, caching, and visibility handling live
in separate modules. Decoded window audio is shared with a 64 MiB cache budget;
up to 40 generated images remain in memory. Failures allow manual retry and do
not block ordinary playback. Signed candidate URLs are never persisted.

Pagination cursors use the service's v2 contract. Older/invalid cursors return
400 and require restarting from the first batch; cursors are opaque to the
frontend. The retry control offers a first-batch restart after cursor rejection.
Every page load starts fresh: names, answers, pending submissions, annotator IDs,
and cursors stay in memory only. Old annotation drafts and catalog storage keys
are removed on startup. Catalog and media caches are reused within the current
page only; reloading clears application caches.

## Annotation submission

The annotator UUID is generated per page load and stays in memory. Display names
are collected separately and trimmed when building a submission. Names are not
identifiers. `annotation-submission.mjs` builds the request:

```json
{
  "submission_id": "UUID",
  "annotator_id": "UUID",
  "annotator_name": "Maria",
  "annotations": [{
    "candidate_id": "candidate ID",
    "window_id": "window ID",
    "query_species_id": "species ID",
    "response": "present"
  }]
}
```

Yes/No/Unsure map to present/absent/unsure. Only changed, answered candidates in
the current batch are sent, up to 20. Draft answers across other species/batches
remain available within the page. The exact pending payload is retained in
memory for manual retries. Reloading discards all unsubmitted answers.

`POST /annotations` must return the matching `submission_id` and complete
`accepted_count` before answers are marked submitted and Next is enabled.
Submission locks the name, answers, and navigation; duplicate clicks are disabled.
A pending failure retains its original payload and name for manual retry. Editing
it requires explicitly abandoning that retry payload.

The backend is append-only: submission IDs group rows but are not idempotency
keys. A network failure, invalid receipt, or 503 has an uncertain outcome.
Manual retry can append duplicates, so writes are never automatically retried.
A 401 occurs before insertion and asks the annotator to log in again.

## Demo mode and local use

Demo species/references use the CSV and static media. Five bundled candidates
are reused for each species. Demo saving is disabled; answers are memory-only.
The exported legacy `buildSubmission` helper describes demo source metadata,
not the cloud POST contract.

Serve the repository root and open `/docs/annotate.html`. Bucket CORS must permit
frontend JavaScript audio reads for local spectrogram generation. Direct signed
image/audio requests remain separate from the authenticated API transport.
