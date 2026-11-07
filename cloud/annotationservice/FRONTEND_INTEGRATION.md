# Frontend handoff: candidates and annotation submission

## Task

First adopt [REVIEWER_ACCESS.md](REVIEWER_ACCESS.md): all species, reference,
candidate, and annotation API calls now need a four-hour reviewer bearer token.
The shared-password gate is deployed. Add the login flow below before making
any of the existing species, reference, candidate, or annotation requests.

Connect the annotation screen to the deployed candidate API. Species selection
and reference media are already integrated; use those existing components.
Inspect the current candidate provider, player, batch navigation, and answer
state before adapting this API into the frontend's models.

Candidate reads are deployed. `POST /annotations` is now implemented in source
and requires the schema migration and redeployment described in
[ANNOTATION_SUBMISSIONS.md](ANNOTATION_SUBMISSIONS.md). Use that document for
the request contract, retry handling, and frontend submission implementation.
There is still no saved-annotation read endpoint. Do not present locally saved
answers as successfully uploaded to BigQuery.

## Shared-password login

1. Show a password form before loading protected data. Use a password input and
   submit the entered value to the endpoint below without trimming it. Obtain
   the shared password from the project maintainer; never hardcode it in source,
   build variables, documentation, or the public GitHub Pages bundle.
2. On success, clear the password input and retain `access_token` and
   `expires_at` in memory or namespaced `sessionStorage`. Do not persist the
   password. Tokens should not go in localStorage, URLs, logs, or saved drafts.
3. Update the shared API transport used by **all four** routes: `/species`,
   `/references`, `/candidates`, and `/annotations`. Include
   `Authorization: Bearer <access_token>` on every request to these routes.
   Existing species and reference integrations need this change too.
4. Tokens expire four hours after login. On a protected request's 401, clear the
   session and reopen the password form while preserving answers, species,
   pagination, annotator name/UUID, and the pending submission payload. After
   login, retry reads or let the user resume submission. Do not automatically
   retry writes after a network error or 503: they may already have succeeded.
5. Add Logout to clear the token. Login 401 means an incorrect password; login
   503 means a backend configuration/service issue. Show distinct messages and
   avoid repeated automatic login attempts.

```http
POST https://annotationservice-167659324212.us-south1.run.app/auth/login
Content-Type: application/json

{"password": "value-entered-in-the-password-form"}
```

A successful response contains `access_token`, `token_type: "Bearer"`,
`expires_in: 14400`, and `expires_at`. Use the `reviewerFetch` helper in
[REVIEWER_ACCESS.md](REVIEWER_ACCESS.md#minimal-api-transport-example), or adapt
the existing transport to the same behavior. Its name is just a frontend helper;
it is unrelated to the backend's `ANNOTATOR_*` secret names.

Send this token **only to the Cloud Run API**, never to signed GCS audio/image
URLs. Media requests keep using their existing signed URLs without an added
Authorization header. No Google login, cookies, or `credentials: "include"`
are needed. Add `<meta name="robots" content="noindex, nofollow">` to the
annotation page; the static page itself remains public.

## Deployed API

Base URL:

```text
https://annotationservice-167659324212.us-south1.run.app
```

Use the existing API configuration. No Google login or credentials are needed,
but enter the shared reviewer password via /auth/login and include its bearer
token. Ordinary cross-origin fetch is supported with the Authorization header.

```http
GET /candidates?species_id=DENEBR&count=15
```

| Parameter | Meaning |
|---|---|
| `species_id` | Required single species ID from the existing selector |
| `count` | Optional integer from 1 to 20; default 15 |
| `cursor` | Optional opaque token returned as `next_cursor` by the prior page |

```json
{
  "candidates": [
    {
      "candidate_id": "test_candidate_DENEBR_1",
      "window_id": "test_window_DENEBR_1",
      "query_species_id": "DENEBR",
      "created_at": "2026-11-05T00:00:00+00:00",
      "score": 0.1,
      "recording_id": "test_recording_DENEBR",
      "audio_url": "https://storage.googleapis.com/...?X-Goog-...",
      "start_seconds": 0,
      "end_seconds": 5,
      "recording_start_seconds": 725,
      "recording_end_seconds": 730,
      "spectrogram_url": null
    }
  ],
  "next_cursor": "opaque-token",
  "urls_expire_at": "2026-11-05T12:15:00+00:00"
}
```

The example shows one sample, not the full fixture response. Use `candidate_id`
as the component and answer-state key. Preserve `window_id` and
`query_species_id` in the data model and send them with the candidate ID and
answer on submission. The backend no longer looks these relationships up. Map snake_case into the existing
frontend naming convention in one provider adapter.

`score` is a finite review-priority value in `[0, 1]`, not a presence probability
or signed logit. Lower values receive review priority. Preserve the server's
ascending `(score, candidate_id)` order; creation time only establishes the
pagination cutoff. The initial multiclass model uses `abs(2 * p - 1)` for a
species' softmax probability `p`, so a score near zero prioritizes predictions
near 0.5. Treat this as a ranking heuristic, not calibrated presence confidence.

## Batch state and navigation

1. On species selection, request the first batch without a cursor. Show a loading
   state, then render the returned candidates.
2. Retain the request's species, count, and input cursor alongside the response's
   `next_cursor` and `urls_expire_at`. For the first batch the input cursor is absent.
3. On Next batch, send `next_cursor` with the same species and count. JSON null
   means the end of the queue; disable Next and show a completion state. Do not
   accidentally omit a null cursor and restart the first page.
4. Preserve unsent answers keyed by candidate ID. Do not discard them on Next or
   species change. Send `POST /annotations` with a stable submission UUID and
   advance only after its receipt confirms success. Retain the exact payload
   and submission ID across timeouts, reloads, and retries.
5. Reset pagination on species change. Abort obsolete requests or ignore stale
   results, including responses arriving after a newer species selection. Disable
   duplicate Next requests while a batch is loading.
6. Distinguish an empty queue from a failed request. Offer retry for failure,
   preserving the current batch and answers.

The cursor is species-bound. Treat it as opaque and use URLSearchParams to encode
it. The first request fixes a creation-time cutoff carried by subsequent cursors;
newer candidates appear when starting a fresh session. This is not a database
snapshot: backdated inserts, edits, and deletes may still affect pages.

Pagination cursors are now version 2 and continue after the last `(score,
candidate_id)` pair. Old version-1 cursors from creation-time ordering are
rejected with HTTP 400 and an instruction to restart from the first batch.
Preserve any pending answer payload while resetting navigation; never silently
discard drafts because their old cursor has expired. Replacing the candidates
table can end an older pagination session; start a fresh first page to see the
new run.

The API does not reserve candidates for a reviewer or exclude previously reviewed
samples yet. Refreshing from the first page can show the same candidates again.

## Audio windows and spectrograms

`audio_url` points to the window's own five-second audio file, signed from
`windows.audio_gcs_uri`. Candidate queries join only candidates and windows;
they no longer depend on recording-level audio or a recordings-table join.

Playback bounds are always `start_seconds: 0` and `end_seconds: 5`. Play and
generate spectrograms over that standalone clip. The original recording offsets
are returned separately as required `recording_start_seconds` and
`recording_end_seconds` metadata; they must span five seconds and must never be
applied to playback or spectrogram cropping. For example, a recording window
at `[725, 730]` plays its returned audio at `[0, 5]`, not `[725, 730]`.

`spectrogram_url` can be null. Show an unavailable state, or use an existing
client-side spectrogram renderer on this exact audio interval. Never render an
image with a null source or substitute a reference spectrogram for a candidate.
The current fixture intentionally has no candidate images.

Signed media URLs expire after 15 minutes. Use the full returned URL unchanged.
Before using media with less than one minute remaining, replay the current
batch's **input cursor**, species, and count to refresh URLs without advancing.
Preserve answers by candidate ID. The first page is refreshed without a cursor;
its membership can change, so reconcile by ID rather than by array position.
On media failure, allow one refresh/retry, then show a manual retry state.

Keep signed URLs out of persistent storage and logs. A draft can retain IDs,
answers, and navigation state; fetch fresh URLs when resuming it.

Plain audio/image elements can load signed media directly. JavaScript fetch,
Web Audio decoding, or canvas access requires bucket CORS for the frontend
origin, separately from API CORS. Do not use `mode: "no-cors"` as a workaround.
Verify media behavior in the actual frontend, not just by opening the URL.

## Request helper

Adapt to the existing frontend transport and error conventions:

```javascript
async function loadCandidates({ speciesId, count = 15, cursor, signal }) {
  const query = new URLSearchParams({
    species_id: speciesId,
    count: String(count),
  });
  if (cursor != null) query.set("cursor", cursor);
  const response = await reviewerFetch(`/candidates?${query}`, {
    signal,
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Candidate request failed (${response.status})`);
  }
  return response.json();
}
```

Known species with an empty queue returns 200 with `candidates: []` and
`next_cursor: null`. Unknown species returns the same successful empty result;
the backend does not separately check species existence. Invalid counts, repeated
parameters, malformed cursors, and a cursor for another species return 400.
Backend query/signing failures return 500. Error responses contain an `error`
string. Do not silently substitute bundled candidates after an API failure.

## Reviewer identity and submission

No individual account is required; the shared password controls API access.
Persist a browser-generated UUID in localStorage as the
anonymous reviewer ID, and collect the reviewer's display name separately.
Names are not unique identifiers. Clearing browser storage loses this identity.
Neither the name nor reviewer ID is needed in candidate GET requests.

The UUID maps to `annotator_id` and the required display name maps to
`annotator_name`. Send 1–20 distinct candidate answers per batch. The name is
stored alongside each answer; it is self-reported and not an authentication
credential. A changed name applies to new batches, not pending retries.

Follow [ANNOTATION_SUBMISSIONS.md](ANNOTATION_SUBMISSIONS.md) for the complete
POST contract and helper. Repeated submissions append new rows, even with the same submission ID.
There is no 409 duplicate/conflict behavior. On a network failure or 503, retain
the payload, show an uncertain-outcome state, and offer manual retry; another
request may append duplicates. Keep the ID to support downstream cleanup. Shared rate limiting
and already-reviewed candidate filtering are deferred in this BigQuery-only
implementation.

## Acceptance checks

- Before login, no protected API requests run. A valid password unlocks species,
  references, candidates, and submission through the shared authenticated transport.
- Wrong passwords show a login error. Expired sessions reopen login without
  losing drafts or advancing a batch; logout clears the token.
- Neither the password nor access tokens appear in source, URLs, or logs, and
  GCS media requests do not receive the API Authorization header.
- Selecting a species loads candidates through `/candidates` while the existing
  species selector and reference panels continue working.
- Candidates load in ascending `(score, candidate_id)` order, including ties
  and score zero. Pagination with a small count visits each candidate once in
  an unchanged dataset and ends with a null cursor.
- A species with no candidates has a clear empty state. Fixture counts/species
  are not application limits to hardcode; old fixtures are cleared by the reset.
- Reusing a cursor returns the same candidate IDs in the unchanged fixture.
- A candidate with nonzero recording offsets plays and generates a spectrogram
  from its standalone audio at zero through five seconds. Recording offsets
  are retained as metadata only.
- Old version-1 cursors return a clear restart instruction without losing drafts.
- Missing images, API failures, media failures, and expired URLs have clear states.
- Switching species quickly cannot display an older selection's batch.
- Refreshing media and navigating batches preserve unsent answers.
- Include window_id and query_species_id in every submitted answer. Replaying
  a batch appends another set of rows; downstream cleaning handles duplicates.
- A failed submission retains the exact pending payload and never advances Next.
- Names and UUIDs stay separate; pending retries retain the original name.
- No fake submission success, Google credentials, direct BigQuery calls, or local
  candidate-server requests are introduced.

The source test suite covers score pagination, tied/zero scores, cursor replay,
old/invalid cursors, standalone window audio, and authentication/submission
regressions without cloud calls. Verify deployed reads after reset/redeployment;
browser playback and generated spectrograms still need frontend verification
with published window audio.

Backend source: `google_cloud/cloud_run_functions/annotationservice/main.py`.
Annotation targets now use `/species?has_audio=true&has_candidates=true`.
Comparison references continue using `/species?has_audio=true`, including species
without candidates. Candidate availability is cached per page instance, not
across visits. Deploy the updated catalog API before using this filter.
