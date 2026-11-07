# Frontend adoption: shared reviewer password

Status: backend deployed; frontend adoption required. Live checks confirm that
unauthenticated data requests return 401, incorrect passwords return 401, and
CORS preflight allows Authorization. Successful login and authenticated flows
still need verification with the frontend.
The public GitHub Pages site stays at https://osaconservation.github.io/OsaAnurans/.
Cloud Run protects data access with a shared password and four-hour signed
bearer tokens. No user account or Google login is required.

## API contract

Base URL: `https://annotationservice-167659324212.us-south1.run.app`

```http
POST /auth/login
Content-Type: application/json

{"password": "password-entered-by-the-reviewer"}
```

Successful response, HTTP 200:

```json
{
  "access_token": "signed-token",
  "token_type": "Bearer",
  "expires_in": 14400,
  "expires_at": "2026-11-06T02:00:00+00:00"
}
```

Send the token on every species, reference, candidate, and annotation API call:

```http
Authorization: Bearer <access_token>
```

Protected routes: GET /species, GET /references, GET /candidates, and
POST /annotations. Query parameters and response bodies are unchanged. The
single-insert annotation payload still requires window_id and query_species_id.
OPTIONS preflight and POST /auth/login are public. Other paths/methods return
404. Missing, invalid, or expired tokens return 401 before parsing data requests,
opening cloud clients, running BigQuery jobs, or signing media URLs.

Login errors: wrong password returns 401; malformed JSON/fields return 400;
non-JSON content type returns 415; bodies over 2 KiB return 413. Passwords are
case-sensitive and whitespace is significant. Do not trim the entered password.
Missing/weak backend secret configuration returns 503, never open access.
All JSON responses use Cache-Control: no-store and retain CORS headers.

## Frontend flow

1. Show the password gate before loading species or other protected data.
2. POST the entered password to /auth/login. Clear the password input afterward;
   do not persist the password or embed it in frontend source/configuration.
3. Store the returned token and expires_at in memory or namespaced sessionStorage
   (for example osa.reviewer.session). sessionStorage supports reloads within
   the tab; it is readable by JavaScript on the same origin. Keep tokens out of
   localStorage, URLs, logs, analytics, and persisted annotation drafts.
4. Add Authorization only to requests sent to the API base URL. On 401, clear
   the token and reopen the password gate. Preserve species selection, cursors,
   reviewer UUID/name, pending answers, and pending submission IDs.
5. After login succeeds again, reload reads or let the user resume submission.
   A 401 is safe to resubmit after login because authentication runs before the
   insert. A 503/network failure remains ambiguous and retrying may append
   duplicates; do not automatically replay those writes.
6. Disable duplicate login/submit clicks. Show password errors separately from
   a 503 backend-configuration/service error. Avoid infinite retry/login loops.
7. On logout, discard the token and stop further protected requests. Decide
   explicitly how to retain local drafts rather than silently losing answers.

Token lifetime is four hours from password entry; it is not extended by activity.
There is no refresh endpoint. Re-enter the password when it expires. Use
expires_at to warn or reopen the gate, but rely on the API's 401 for enforcement.
The bearer token grants reviewer access; annotator_id and annotator_name still
describe self-reported browser identity and display name, independently.

## Minimal API transport example

Integrate this with the existing request layer and UI state. The example uses
memory; add namespaced sessionStorage persistence if reload recovery is desired.

```javascript
const API_BASE = "https://annotationservice-167659324212.us-south1.run.app";
let reviewerSession = null;

async function loginReviewer(password) {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
    cache: "no-store",
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Unable to sign in");
  reviewerSession = result;
}

async function reviewerFetch(path, options = {}) {
  // Call only for known API routes, never for arbitrary/media URLs.
  if (!reviewerSession) throw new Error("Reviewer password required");
  const headers = new Headers(options.headers);
  headers.set("Authorization", `Bearer ${reviewerSession.access_token}`);
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    cache: "no-store",
  });
  if (response.status === 401) {
    reviewerSession = null;
    // Signal the UI to reopen the password gate and preserve draft/batch state.
    throw new Error("Reviewer session expired; enter the password again");
  }
  return response; // Existing callers must still check response.ok.
}
```

Examples: reviewerFetch('/species?has_audio=true') and
reviewerFetch('/annotations', {method: 'POST', headers: {'Content-Type':
'application/json'}, body: JSON.stringify(savedSubmission)}).

## Media and GitHub Pages

Keep direct audio/image requests using the existing signed GCS URLs. Do not send
the reviewer Authorization header to storage.googleapis.com. Browser spectrogram
generation uses ordinary fetch(audio_url), with bucket CORS already configured.
Media URLs last 15 minutes independently of the four-hour reviewer token. While
the reviewer token is valid, refresh expired media URLs by replaying the current
API read. Already-issued GCS URLs remain usable until their own expiry after
logout or token expiry; they are not individually revoked by this gate.

The static page source remains public. Add this to the annotation page's HTML
head to discourage indexing, while keeping authorization on Cloud Run:

```html
<meta name="robots" content="noindex, nofollow">
```

API CORS now allows Authorization and Content-Type. No cookies or
credentials: 'include' are required. The API's public Cloud Run invocation
setting must remain enabled: the shared reviewer token is application auth,
not a Google IAM identity token.

## Backend deployment in Cloud Console

1. In project osaanurans, enable Secret Manager if necessary. Create two secrets
   with distinct random values: one reviewer password (12-1024 characters),
   and a separate signing secret (at least 32 UTF-8 bytes).
   Use a long random password for the small reviewer group. Generate values
   privately with a password manager or Python secrets.token_urlsafe(48);
   never add real values to this repository or send them in this handoff.
2. Grant the function runtime service account Secret Manager Secret Accessor
   on these two secret resources. Existing BigQuery/GCS roles remain unchanged.
3. In the Cloud Run revision settings, expose the password secret as environment
   variable ANNOTATOR_PASSWORD and the signing secret as ANNOTATOR_TOKEN_SECRET.
   Pin explicit secret versions so all instances of the revision agree.
4. Deploy main.py, auth.py, settings.py, catalog.py, candidates.py, media.py,
   annotate.py, and the updated requirements.txt. PyJWT is the only new dependency.
   Keep the function entry point annotationservice and unauthenticated Cloud Run
   invocation enabled. Do not add an auth-disabled fallback configuration.
5. Update the frontend request layer before expecting existing data loading to
   work: unauthenticated data requests will now return 401. Deploy frontend/backend
   changes together, and route traffic to the intended backend revision.

These checks are local and need a deployed verification after setup:

- A request without a token returns 401; OPTIONS still returns 204.
- Wrong password returns 401; the correct password issues a four-hour token.
- With that token, species, references, candidates, and submissions work.
- Invalid/expired tokens are rejected before cloud calls.
- Password expiry in the frontend preserves drafts and resumes after login.
- API and GCS audio fetches work from the actual GitHub Pages origin.

There is no server-side session database. Logout clears the browser's token;
it cannot revoke a copied token. To invalidate all sessions, rotate the signing
secret and deploy a new revision with all traffic shifted to it. Changing the
password alone stops new logins with the old password but does not invalidate
existing tokens. Shared rate limiting and login-attempt throttling remain
deferred; this gate deters ordinary crawlers rather than proving human identity.

References: [Cloud Run secrets](https://docs.cloud.google.com/run/docs/configuring/services/secrets),
[PyJWT validation](https://pyjwt.readthedocs.io/en/latest/api.html), and
[Google noindex guidance](https://developers.google.com/search/docs/crawling-indexing/block-indexing).
