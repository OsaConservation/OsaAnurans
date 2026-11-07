# Annotation service

A Python **Cloud Run function** that serves species, reference media, and candidates
from BigQuery and accepts annotation batches. Its HTTP entry point is
`annotationservice` in `main.py`; submission handling lives in `annotate.py`.

For the frontend implementation task, read
[FRONTEND_INTEGRATION.md](FRONTEND_INTEGRATION.md), which now focuses on loading
candidate batches for annotation, pagination, playback, and acceptance checks.
Species and reference integration is already complete.

All data endpoints now require a four-hour reviewer bearer token. Read
[REVIEWER_ACCESS.md](REVIEWER_ACCESS.md) for the shared-password endpoint,
frontend adoption instructions, and required Cloud Run secret configuration.
This change needs deployment together with the frontend password gate.

## Source layout

| File | Responsibility |
|---|---|
| `main.py` | HTTP routing, response headers, and error handling |
| `auth.py` | Shared-password login and four-hour bearer-token verification |
| `settings.py` | Shared settings, BigQuery client, and bounded read queries |
| `catalog.py` | Species/reference validation and reads |
| `candidates.py` | Cursor validation, candidate queries, and batch responses |
| `media.py` | Credentials and per-response signed media URLs |
| `annotate.py` | Basic request checks and one parameterized annotation INSERT |

Requests require Authorization: Bearer. Species rows include `annotation_counts`
with `positive`, `negative`, `unsure`, and `total` integers. Counts include seed
annotations and every saved vote, including repeated submissions. All species
are returned by the unfiltered endpoint, with zeros for species without votes.
Annotation payloads include
window_id and query_species_id; repeated submissions append duplicates for
downstream cleaning. See ANNOTATION_SUBMISSIONS.md before updating the frontend.

## Cloud configuration

- Project: `osaanurans`
- BigQuery dataset: `osafrogdataset`
- BigQuery location: `us-south1`
- Read tables: `species`, `reference_media`, `candidates`, `windows`, `annotations`
- Write table: `annotations`

The runtime service account needs **BigQuery Job User**
(`roles/bigquery.jobUser`) on the project and **BigQuery Data Viewer**
(`roles/bigquery.dataViewer`) on the dataset. It uses its attached service
identity; no credential JSON belongs in the source directory.
For submissions, also grant **BigQuery Data Editor** (`roles/bigquery.dataEditor`)
on the `annotations` table. Dataset-wide write access is not required. The
service uses its attached runtime identity for writes.

For private media URLs, also configure the following:

1. Enable the **IAM Service Account Credentials API** in project `osaanurans`.
2. Find the function's runtime service account in its Cloud Run security settings.
3. On bucket `osa-frog-data`, grant that account **Storage Object Viewer**
   (`roles/storage.objectViewer`). Keep the bucket private.
4. In **IAM & Admin > Service Accounts**, open that same account's permissions
   and grant **Service Account Token Creator** (`roles/iam.serviceAccountTokenCreator`)
   to the runtime account itself. This is a role on the service account resource,
   not a project-wide grant. It allows IAM to sign URLs without a key file.

By default the signing account is the attached runtime account. Optionally set
the environment variable `SIGNING_SERVICE_ACCOUNT` to a different service account
email; that signing account needs bucket read access, and the runtime account
needs Token Creator on it. Changes to IAM can take time to propagate.

## Endpoints

`POST /auth/login` takes a JSON password and issues a four-hour bearer token.
`OPTIONS` remains public. All GET endpoints below and POST /annotations require
that token; missing/invalid/expired tokens return 401 before cloud access.
There is no BigQuery query or server-side session store for login/token checks.

### `GET /species`

Returns species sorted by scientific name:

Optional filter: `GET /species?has_audio=true` returns only species with at
least one reference-media row containing a nonempty `audio_gcs_uri`. Omit the
parameter or use `has_audio=false` to return all species. Supply the parameter
once, with the exact value `true` or `false`; invalid values return HTTP 400.
This checks catalog metadata, not object existence in GCS, and does not require
a spectrogram. Candidate recording audio does not qualify as reference audio.
Annotation counts are included with or without these filters.

`has_candidates=true` additionally requires a candidate joined to a window with
nonempty audio URI. Combine both filters with
`/species?has_audio=true&has_candidates=true` for annotation targets. Omitted or
`false` applies no candidate filter. As with `has_audio`, invalid or repeated
values return HTTP 400. Previously answered candidates remain available under
the existing candidate API; this filter does not exclude annotator answers.

```json
{"species": [{"species_id": "AGACAL", "scientific_name": "Agalychnis callidryas", "common_name": "Red-eyed treefrog"}]}
```

### `GET /references?species_id=AGACAL&species_id=DENEBR`

Supply 1–52 nonempty `species_id` parameters. Duplicates are removed and the
query uses a parameterized STRING array. Missing/invalid lists return HTTP 400.
Unknown IDs and species without reference media return empty lists.

```json
{
  "references": {
    "AGACAL": [
      {
        "reference_id": "AGACAL_1",
        "audio_url": "https://storage.googleapis.com/osa-frog-data/reference_catalog/AGACAL_1.mp3?X-Goog-...",
        "spectrogram_url": "https://storage.googleapis.com/osa-frog-data/reference_catalog/AGACAL_1.webp?X-Goog-..."
      }
    ],
    "AGAANN": []
  },
  "urls_expire_at": "2026-11-05T12:15:00+00:00"
}
```

The response now returns `audio_url` and `spectrogram_url` instead of GCS URI
fields. Use them directly as the `src` of audio/image elements. Missing
spectrograms return null. BigQuery keeps permanent `gs://` URIs; only responses
contain signed URLs. The browser fetches media directly from GCS.

URLs allow GET access for 15 minutes; `urls_expire_at` is their UTC expiration.
Fetch `/references` again before using expired URLs. Successful API responses
use `Cache-Control: no-store`. Anyone with a URL can read that object until it
expires, and a public endpoint allows anyone to request URLs. No object existence
check is made during signing; a missing file will fail when the browser fetches it.
If the frontend uses JavaScript fetch/Web Audio or canvas access, configure GCS
bucket CORS for the frontend origin as well; function CORS only covers API calls.

Endpoints allow cross-origin requests (`Access-Control-Allow-Origin: *`),
including the Authorization header.
`OPTIONS` returns HTTP 204; unsupported paths/methods return 404. Query failures
return 500 with details recorded in Cloud Logging. Queries have a 100 MB
maximum-bytes-billed limit per read query job. The parameter-only annotation INSERT
has a 100 MB cap and does not scan existing tables. These limits cap billed query
work, not uploaded audio size.
Shared rate limiting is deferred;
there is no Firestore dependency.

### `GET /candidates?species_id=DENEBR&count=15`

Returns a page of candidates with signed full-recording audio URLs and window
playback offsets. Required: one nonempty `species_id`. Optional: `count`, an
integer from 1 to 20 (default 15), and `cursor` from a previous response.
Repeated parameters and invalid counts/cursors return 400. Unknown species and
species with no candidates both return 200 with an empty list and a null cursor.
Each batch uses one query; there is no separate species-existence lookup.

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
      "recording_start_seconds": 120,
      "recording_end_seconds": 125,
      "spectrogram_url": null
    }
  ],
  "next_cursor": "opaque-continuation-token-or-null",
  "urls_expire_at": "2026-11-05T12:15:00+00:00"
}
```

Pass `next_cursor` as `cursor` to fetch the next page for the same species.
An actual exhausted response uses JSON null, not the string above. Ordering is
ascending `(score, candidate_id)`; smaller scores receive review priority.
The first request fixes a creation-time cutoff, carried through subsequent
pages. Candidates created after that cutoff appear in a new session. This is
not a database snapshot: backdated inserts, deletes, or edits can change pages.
Keep published candidate IDs, scores, and creation times immutable. Cursor format is v2; old cursors return 400 and require restarting from the first batch. Cursor tokens are
validated, species-bound continuation data, not signed authorization tokens.
All candidate reads require a reviewer token and remain bounded and parameterized regardless of cursor.

Save the input cursor (or absence of one for the first page) as well as the next
cursor. Replay the input cursor to refresh a batch's expired media URLs. Keep
count unchanged when refreshing, and preserve unsent answers by candidate ID.
URLs expire after 15 minutes, with the same private-bucket and CORS rules as
reference media. Each window has a standalone five-second audio file in
`windows.audio_gcs_uri`. Playback bounds are always `[0, 5]`; original recording
offsets are returned separately as metadata and must not be applied to playback
or local spectrogram generation. Candidate queries no longer join recordings.

No reviewer identity is required to read. For submissions, use a
browser-generated UUID persisted in localStorage as `annotator_id`, and ask for
a display name separately. Names must not serve as unique identifiers. Run the
name-column migration below before enabling submissions. This endpoint does not yet exclude previously
annotated candidates or reserve batches for one reviewer. Different reviewers
can receive the same candidates.

### `POST /annotations`

Accepts 1-20 answers with window_id, query_species_id, response, and optional
candidate_id, plus submission/reviewer UUIDs and a display name. The frontend
supplies relationships; the service only checks request format in Python and
performs a single INSERT. No database lookups or duplicate checks run. Replaying
a request appends new rows. Training data preparation owns cleanup.

See [ANNOTATION_SUBMISSIONS.md](ANNOTATION_SUBMISSIONS.md) for the new payload,
error behavior, and frontend changes. No shared rate limiting is implemented.

## Deploy using the Cloud Console

Open your existing Cloud Run function in project `osaanurans` and edit its source:

1. Use a supported Python runtime, such as Python 3.12.
2. In BigQuery Studio (`us-south1`), run
   [add_annotator_name.sql](../../bigquery/add_annotator_name.sql) on the existing
   annotations table. Existing rows remain unchanged; new API writes require a name.
3. Grant the runtime account Data Editor on `annotations`, as described above.
4. Configure ANNOTATOR_PASSWORD and ANNOTATOR_TOKEN_SECRET using Secret Manager,
   following [REVIEWER_ACCESS.md](REVIEWER_ACCESS.md). Missing configuration fails
   closed. Copy **all seven top-level Python files** and `requirements.txt`
   into the function source. Keep them side by side in the console editor:
   `main.py`, `settings.py`, `catalog.py`, `candidates.py`, `media.py`,
   `annotate.py`, and `auth.py`. Omitting an imported module causes a
   startup failure. Tests and Markdown files are not needed for deployment.
   PyJWT is the new dependency. Existing BigQuery/GCS migration and IAM steps do
   not need repeating; add Secret Accessor for the two injected secret resources.
5. Set the function entry point to **`annotationservice`**.
6. Select a runtime service account with the BigQuery, Storage, and signing roles
   above and deploy.

Use your deployed function base URL followed by `/species` or `/references`.
For example: `https://YOUR_FUNCTION_URL/references?species_id=AGACAL&species_id=DENEBR`.
Cloud Run invocation authentication is configured separately from CORS; retain
the invocation settings you chose for your existing frontend integration.

See Google's [Cloud Run function source requirements](https://docs.cloud.google.com/run/docs/write-functions).

## Run locally

From this directory, using a Python environment with local Application Default
Credentials:

```powershell
python -m pip install -r requirements.txt
gcloud auth application-default login
$env:SIGNING_SERVICE_ACCOUNT = 'YOUR_SIGNING_ACCOUNT@osaanurans.iam.gserviceaccount.com'
# Also set ANNOTATOR_PASSWORD and ANNOTATOR_TOKEN_SECRET privately; never commit them.
functions-framework --target annotationservice --port 8080
```

POST the password to `http://localhost:8080/auth/login`, then include the returned
bearer token on calls to `/species`, `/references`, `/candidates`, and `/annotations`.

For local signing, your logged-in user also needs Service Account Token Creator
on the configured signing account. `/species` needs no signing setup.

## Verification and troubleshooting

Run `python -m unittest discover -s tests -v` from this directory for mocked
HTTP, signing, format-check, and append-only write tests. Cloud calls use mocks;
these tests do not contact BigQuery/GCS/IAM. Follow the manual submission
checks in [ANNOTATION_SUBMISSIONS.md](ANNOTATION_SUBMISSIONS.md) after deployment.

After deployment, log in and fetch `/references?species_id=AGACAL` with its token, then open the returned
`audio_url` and `spectrogram_url` in your browser. To verify privacy, the same
object URL without its signature should fail without Google authentication.

- A `signBlob` permission error in Cloud Logging means the caller needs Token
  Creator on the signing service account, or the IAM Credentials API is disabled.
- A signed media URL returning 403 can indicate the signing account lacks Storage
  Object Viewer, the URL expired, or its query parameters were modified.
- Signing does not change bucket IAM, frontend authentication, or rate limiting.

See [signed URLs](https://docs.cloud.google.com/storage/docs/access-control/signed-urls)
and [IAM signing permissions](https://docs.cloud.google.com/iam/docs/reference/credentials/rest/v1/projects.serviceAccounts/signBlob).
