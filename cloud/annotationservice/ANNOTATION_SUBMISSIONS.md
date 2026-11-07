# Submit annotation batches

Status: the simplified append-only implementation needs redeployment. Deploy
all seven Python files listed in README.md, plus requirements.txt. No schema
change is required if annotator_name was already added. PyJWT is the new auth
dependency. First adopt [REVIEWER_ACCESS.md](REVIEWER_ACCESS.md): this endpoint
requires a four-hour bearer token issued after entering the shared password.

## Contract

POST /annotations with Content-Type: application/json:

Include Authorization: Bearer <access_token>. Unauthorized requests return 401
before any insert and can be resubmitted after login. Network failures and 503
responses remain ambiguous, as explained below.

```json
{
  "submission_id": "a8ff083c-d6a1-43f9-bcf2-e52bf19264ba",
  "annotator_id": "0c3ab387-866c-4e9a-b15f-75c21d0d9a76",
  "annotator_name": "Maria",
  "annotations": [
    {
      "candidate_id": "test_candidate_DENEBR_1",
      "window_id": "test_window_DENEBR_1",
      "query_species_id": "DENEBR",
      "response": "present"
    }
  ]
}
```

The frontend must now include window_id and query_species_id from each loaded
candidate. The previous candidate-ID-only payload is no longer sufficient.
candidate_id is optional (omitted or null) for annotations outside the candidate
queue. The service trusts the submitted relationships and does not query their
existence or consistency. Training/evaluation data preparation owns that cleanup.

The browser reviewer UUID is persisted in localStorage. Collect a display name
separately; names and browser UUIDs are self-reported. submission_id groups a
batch for later inspection and cleanup; it is not an idempotency key.

Basic format checks still run in Python, without querying BigQuery: valid UUIDs,
a nonempty name of up to 100 characters, 1?20 answers, printable IDs, and responses
present/absent/unsure. Unknown fields, duplicate JSON keys, invalid text, and bodies
over 16 KiB are rejected. Duplicate answers within a batch are accepted.

The successful response confirms this request's inserted row count:

```json
{
  "submission_id": "a8ff083c-d6a1-43f9-bcf2-e52bf19264ba",
  "accepted_count": 1
}
```

## Storage behavior

Each request executes one parameterized INSERT using the submitted JSON array.
There are no source-table scans, joins, temporary tables, SQL assertions,
previous-submission checks, or application-managed job-retry chains. BigQuery
generates a new annotation UUID per row and assigns a UTC server timestamp.
The single INSERT is atomic, without a multi-statement transaction.

Repeating a request, even with the same submission ID, appends another set of
rows with new annotation IDs. Changed answers or names under the same submission
ID are also accepted. Relationship validation and duplicate cleaning happen
later on the training side. No shared rate limiting or reviewed-candidate
filtering is implemented.

A 503 or network timeout means the outcome is uncertain: the write may already
have completed. Retrying is allowed, but can append duplicates. Preserve the
original submission ID on a retry to aid downstream grouping and cleanup.
Do not claim exactly-once delivery or expect a 409 conflict response.

## Frontend implementation

Construct a batch once when the user clicks Submit:

```javascript
const submission = {
  submission_id: crypto.randomUUID(),
  annotator_id: reviewerId,
  annotator_name: reviewerName.trim(),
  annotations: answers.map(({ candidate, response }) => ({
    candidate_id: candidate.candidate_id,
    window_id: candidate.window_id,
    query_species_id: candidate.query_species_id,
    response,
  })),
};

const response = await reviewerFetch(`/annotations`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(submission),
  cache: "no-store",
});
const receipt = await response.json();
if (!response.ok) throw new Error(receipt.error || "Submission failed");
```

Adapt the fields if the frontend uses a camelCase adapter. Disable duplicate
Submit clicks and advance the cursor only after a 200 receipt confirms the
matching submission ID and expected answer count. Preserve drafts after failure
and offer manual retry with an uncertain-outcome state rather than an automatic
retry loop. Do not upload signed media URLs in annotation payloads.

| Status | Meaning |
|---|---|
| 200 | This INSERT completed; accepted_count is its row count |
| 400 | Invalid JSON, required fields, or answer format |
| 401 | Missing/invalid/expired reviewer token; enter the shared password again |
| 413 | Body exceeds 16 KiB |
| 415 | Content type is not JSON |
| 503 | Unable to confirm the write; retry may append duplicates |

Responses have CORS headers and Cache-Control: no-store. Retry-After is exposed
for service failures, but does not imply an idempotent retry guarantee.

## Deployment and verification

1. Deploy main.py, auth.py, settings.py, catalog.py, candidates.py, media.py,
   annotate.py, and requirements.txt. Configure secrets and log in as described
   in REVIEWER_ACCESS.md. annotation_sql.py is no longer used and was removed.
2. Update the frontend to send window_id and query_species_id for each answer.
3. Submit two fixture answers with a fresh submission UUID. Expect count 2.
4. Replay once. Expect count 2 again and FOUR total rows for that submission.
5. Submit an unknown window/species ID if testing dirty data acceptance; it is
   stored as supplied. Use a clearly marked test reviewer name.
6. Inspect rows with ../../bigquery/verify_annotations.sql. Duplicate annotation
   UUIDs should be zero, while repeated candidate IDs are expected after replay.
7. Check BigQuery job statistics to measure actual billed processing. This
   parameter-only INSERT eliminates the earlier multi-statement table scans;
   the 100 MB cap is a ceiling, not a claimed or measured charge.

Local tests use mocked BigQuery jobs. The simplified INSERT still requires a
post-deployment test; earlier live results belong to the previous implementation.
