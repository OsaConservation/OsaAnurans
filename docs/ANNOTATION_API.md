# Annotation frontend data boundary

`annotation-config.js` selects demo or server mode. The server base URL points
to the deployed Cloud Run annotation service. The page is static on GitHub Pages;
Cloud Run enforces reviewer access. The page has `noindex, nofollow` metadata.

## Reviewer access

`reviewer-session.mjs` posts the entered password unchanged to `/auth/login`.
The password is cleared after each attempt. Four-hour bearer tokens remain in
memory and are attached only to the known Cloud Run API routes. Tokens never go
to GCS, localStorage, URLs, or drafts. Refreshing the page requires login again.

A 401 clears the session and shows the gate without discarding answers, reviewer
identity/name, pagination, or a pending submission. Logout hides the workspace,
pauses audio, and clears the token. Reauthentication resumes reads but never
replays a submission automatically. Already-issued signed media URLs retain
their own expiration after logout.

## Catalog and candidates

`ServerProvider` maps API snake_case fields into the frontend model:

- `/species?has_audio=true` supplies the selector and common names.
- `/references` uses repeated `species_id` parameters. All other species with
  audio appear as comparisons. Reference URLs refresh near expiration.
- `/candidates` receives species, count (slider defaults to 5), and an optional
  opaque cursor. The adapter retains input cursor, next cursor, and expiration.

Candidate IDs key answers; window and query-species IDs are retained separately.
A null next cursor disables Next. Failed requests retain the current batch and
retry the failed request. Obsolete responses cannot replace newer selections.

Candidate audio URLs contain full recordings. `candidate-player.mjs` and
`audio-window.mjs` provide clip-relative controls and bounded playback.
Missing spectrogram URLs trigger local generation for the exact window as cards
become visible. Analysis, canvas rendering, caching, and visibility handling live
in separate modules. Decoded recordings are shared with a 64 MiB cache budget;
up to 40 generated images remain in memory. Failures allow manual retry and do
not block ordinary playback. Signed candidate URLs are never persisted.

## Annotation submission

The reviewer UUID is generated once and stored in localStorage. Display names
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
remain available. The exact pending payload, answers, name, and navigation are
saved in localStorage before POST; media URLs, passwords, and tokens are excluded.

`POST /annotations` must return the matching `submission_id` and complete
`accepted_count` before answers are marked submitted and Next is enabled.
Submission locks the name, answers, and navigation; duplicate clicks are disabled.
A pending failure retains its original payload and name for manual retry. Editing
it requires explicitly abandoning that retry payload.

The backend is append-only: submission IDs group rows but are not idempotency
keys. A network failure, invalid receipt, or 503 has an uncertain outcome.
Manual retry can append duplicates, so writes are never automatically retried.
A 401 occurs before insertion and asks the reviewer to log in again.

## Demo mode and local use

Demo species/references use the CSV and static media. Five bundled candidates
are reused for each species. Demo saving is disabled; answers are memory-only.
The exported legacy `buildSubmission` helper describes demo source metadata,
not the cloud POST contract.

Serve the repository root and open `/docs/annotate.html`. Bucket CORS must permit
frontend JavaScript audio reads for local spectrogram generation. Direct signed
image/audio requests remain separate from the authenticated API transport.
