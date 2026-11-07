# Implementation validation

Validated on Windows, Python 3.11.9, bacpipe 1.3.5, using the committed `uv.lock`.

## Automated checks

- 90 tests pass, covering label import, voting, catalog identity/aliases,
  immutable waveform storage, actual bacpipe resampling/windowing, checkpoint
  verification, cache repair, grouped training, artifact reload, all candidate
  pairs, preflight validation, and cloud failures with fake clients.
- Ruff lint and format checks pass.
- Direct and interactive annotation refresh check BigQuery access without creating
  a GCS client. Publication still requires both services.
- Shared JSON writing rejects unsupported/nonfinite values by default, supports
  explicit path stringification, and preserves existing output on failed writes.
- Candidate selection keeps the lowest N per species (default 100), with stable
  score ties and fewer-than-N handling. Fake-cloud tests verify only selected
  windows upload, shared audio uploads once, and retries skip registered windows.
- `uv sync --locked` succeeds and `uv pip check` reports compatible dependencies.
- Tests emit five third-party Python deprecation/future-support warnings.

## Real Grillos smoke test

Command, run from repository root:

```powershell
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli --artifacts osa_anurans_app/artifacts-smoke both --limit-recordings 10
```

Actual Perch v2 ONNX inference ran on CPU, with verified checkpoint checksum.
No fake embedding backend was used for this run.

- Ten source recordings produced 447 standalone five-second WAVs and 447
  aligned 1536-dimensional embeddings.
- The second import reused the first five recordings and generated five more.
- CSV mapping yielded 41 single-label examples; 403 unknown/absent-only windows
  and three multi-positive windows were excluded from training.
- The recording-disjoint split contained 34 training and seven test examples.
- All five species trained: BOAROS, DENEBR, LEPINS, PHYVIT, SCIBOU.
- Generated 2,235 unique candidates: every window times every trained species.
- Independently checked every audio header, window/species coverage, candidate
  uniqueness/ranges, disjoint recording partitions, and reloaded-model scores.
- Some species lack test positives in this small sample; their unavailable
  metrics are reported explicitly. This smoke test does not establish model quality.

Local output run: `artifacts-smoke/runs/20261106T231628Z-4d1db559`.
Generated media, models, embeddings, and download intermediates are gitignored.

## Full-source preflight and remaining scope

The source inventory has 680 WAV files, approximately 24.8 hours of audio, and
270 CSV intervals. One WAV is empty (`sw2/1Jsw2_20261010_192000.wav`), and one CSV
row references unavailable `H75/H75_20230514_184000.WAV`. The strict full-source
preflight stopped before inference and saved its report in `artifacts-preflight`.
The full dataset has not been embedded; fix the inputs or explicitly choose
`--skip-invalid` for that run.

The initial local smoke run made no cloud writes. GPU execution was not tested.

## Window-audio migration and live cloud verification (2026-11-07)

The owner approved clearing test data while keeping species and reference media.
Executed `structs/reset_test_data_window_audio.sql`: six non-catalog tables are
empty, recordings no longer hold an audio URI, and each cloud window requires
`audio_gcs_uri`. Removed the three old GCS test recordings and their folder marker.
Verified all 52 species, 19 reference rows, and 40 reference-folder objects remain
unchanged, including object generations.

Cloud Run revision `annotationservice-00022-2sd` serves all traffic. Its runtime
identity, environment, and secret references match the previous revision.
Candidates now use standalone window audio with `[0, 5]` playback bounds, original
recording offsets as separate metadata, and ascending score/ID pagination.

Live verification exercised actual window audio uploads, idempotent metadata
insertion, candidate staging/replacement SQL, zero scores and score ties across
pages, nonzero recording offsets, signed WAV downloads with browser CORS, legacy
cursor rejection, and annotation submission/receipt. All temporary validation
rows and GCS objects were removed afterward. The final state has zero rows in
deployments/recordings/windows/embeddings/candidates/annotations, leaving the queue
empty until real data is published.

The Cloud Run suite passes 40 tests and the frontend suite passes 18 tests.
Frontend source supports the new contract, a first-batch restart for stale cursors,
and a new draft namespace that does not restore cleared test answers. These
frontend changes are local on `bq-integration`; they have not been pushed to GitHub.

## Full Grillos seed — November 7, 2026

- Removed the 50 dry-run answers/candidates/windows and their seven recording rows.
- Deleted all 50 dry-run window WAVs and reset their local upload flags.
- Verified species, reference rows, and reference object generations were unchanged.
- Imported 679 readable recordings: 17,941 aligned local WAVs and Perch embeddings.
- Skipped the unreadable sw2 WAV and the missing H75 CSV recording, with reports.
- Collapsed duplicate CSV labels into 254 unique window/species presence votes.
- Uploaded only 248 annotated windows, totaling 158,739,840 bytes, from 132 recordings.
- Seeded annotator ID/name `grillos-seed-gt`; candidate IDs are null.
- Retried the seed: counts stayed at 254 votes, with no duplicate annotation IDs/pairs.
- Refreshed 254 answers through the normal BigQuery path; all have local embeddings.
- Trained on 242 single-species examples; six multi-species windows were excluded.
- Split 106/26 whole recordings into 196 training and 46 held-out examples.
- All five species appear on both sides. Accuracy: 88.78% training, 84.78% held-out.
- Saved 50 candidates locally; BigQuery candidates remain empty after dry-run cleanup.
- Run: `artifacts-smoke/runs/20261107T164736Z-d2985fe0`.
- Evidence: `artifacts-smoke/seed-verification.json` and the import/run manifests.
- Python suite: 100 tests passed; lint and formatting passed for 37 Python files.

The removable seed script lives in `scripts/seed_grillos.py`. Normal refresh and
training use BigQuery snapshots and local artifacts, without reading original audio
or the Grillos CSV. Keep local WAVs for later candidate publication.

## Production artifact directory (2026-11-07)

Renamed the populated dataset to `artifacts`; older smoke runs, snapshots,
and preflight reports were removed. Historical paths above describe their
original locations. Retained run: `20261107T165347Z-0bc7081f`.
