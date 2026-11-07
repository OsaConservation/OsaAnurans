# Code review guide

This guide covers the Python workflow in `osa_anurans_app`, from local recordings
to embeddings, training, and cloud publication. Start with the records and data
flow; read the storage details after the rules make sense.

## Where to start

Read these in order:

1. [src/records.py](src/records.py): records passed between stages.
2. [src/config.py](src/config.py) and [config.example.toml](config.example.toml):
   paths, training settings, and cloud destinations.
3. [src/pipeline.py](src/pipeline.py): `import_local`, then `train_local`, then
   `refresh_annotations`. Follow helper functions when needed.
4. [tests/test_pipeline.py](tests/test_pipeline.py): end-to-end examples with
   short real WAVs and a tiny test embedding backend.
5. [src/cli.py](src/cli.py): how commands invoke these operations.

```text
import
  discover/index recordings -> validate CSV -> prepare five-second frames
  -> embed frames and save their WAVs -> register mappings -> snapshot labels

train
  embedding configuration + label snapshot -> select examples
  -> split by recording -> bacpipe training -> save model and metrics
  -> score all windows -> keep lowest N per species -> save candidates.jsonl

publish
  validate saved candidates -> upload WAVs -> insert missing window metadata
  -> stage and validate candidates in BigQuery
  -> replace candidate table in a transaction

refresh
  fetch BigQuery answers -> count votes by window/species IDs
  -> save new snapshot -> update latest snapshot pointer
```

`import`, `train`, and `both` do not publish to the cloud. Interactive training
can explicitly request a BigQuery refresh. `refresh` and `publish` are
cloud actions. `inference` and `add-species` remain reserved.

## Identities and the local catalog

Read [src/audio.py](src/audio.py), then [src/catalog.py](src/catalog.py).

| Record | Identity or lookup | Reason |
| --- | --- | --- |
| Recording | UUID from source-file SHA-256 | Moving a file preserves identity; changing its bytes creates a new one. |
| Window | UUID from recording ID, original sample bounds, and extraction policy | Model changes do not change annotation identities. |
| Embedding | `(window_id, config_id)` maps to `.npy` path and row index | One window can have embeddings from multiple configurations. |
| Label | `(window_id, species_id)` plus vote counts | Multiple species can be present in one window. |
| Candidate | UUID from training run ID, window ID, and species ID | Repeating generation within a run preserves IDs; new runs get new candidates. |

SQLite is an index, not the vector payload store. It connects window IDs to NumPy
rows, WAVs, and upload/registration state. Embeddings stay local.

Check these distinctions carefully:

- Recording hashes use file bytes, not decoded samples. Re-encoding creates a new identity.
- Window bounds use the original sample rate. Model-input WAVs are mono 32 kHz
  and play from zero to five seconds, regardless of source offset.
- The short final window is wrap-padded, not discarded. Its logical end can
  extend beyond the source duration.
- The same prepared frames feed embeddings and saved WAVs. Existing waveform
  content cannot silently change under the same window ID; preprocessing changes
  may require a new extraction policy.
- Path aliases track current files while archived recording metadata remains
  available for historical windows and synchronization.
- Cloud and local records share window IDs; no identity translation is needed.

Review question: can you follow an annotation's window ID to a vector and back
without relying on filename conventions?

## Labels and model assumptions

Read [src/sources.py](src/sources.py), [src/annotations.py](src/annotations.py),
and [src/model/datasets.py](src/model/datasets.py).

**Initial labels:** CSV intervals establish presence for any positive-duration
overlap. Touching a boundary is not overlap. Partial coverage is accepted with
a warning. Repeated intervals for a window/species produce one bootstrap label.
Unlabeled regions are unknown, not absent: the source annotations are not exhaustive.

**Annotator consensus:** count every present and absent answer for each
window/species pair. Ignore unsure votes. Strict majority wins; ties, including
zero votes, are unknown. There is no latest-answer rule or annotator deduplication.

**Snapshots:** store immutable vote counts and, for cloud refreshes, original
answers. Refresh creates a new snapshot rather than merging initial CSV labels
into annotator consensus. Unmatched cloud window IDs are retained and reported,
but cannot train without local embeddings.

**Initial model:** single-label multiclass softmax, using exactly one known
present species per training window. Unknown, absent-only, and multiple-known-
present windows are excluded. At least two classes need training examples.

Exactly one *known* positive does not establish that every other species is
absent. The initial model nevertheless learns competing classes. It does not
provide independent species-presence probabilities. Independent multilabel
classification needs a changed objective and appropriate labels; the annotation
records already support multiple species.

Review question: do the exclusions and warnings make the training population
clear, and do our claims about predictions match these assumptions?

## What bacpipe owns

Read [src/bacpipe_adapter.py](src/bacpipe_adapter.py),
[src/model/train.py](src/model/train.py), and [src/model/model.py](src/model/model.py).

We reuse bacpipe 1.3.5 for audio preparation, Perch inference, `LinearProbe`, and
`train_linear_probe`. Our code supplies window identity, catalog mappings,
recording-level splits, and artifact metadata.

- The checkpoint revision and checksum are pinned. The embedding configuration
  ID captures the model and preprocessing metadata.
- `ProbeDataset` reads explicit vector rows with bounded open memory mappings.
- The seeded split keeps whole recordings together, avoiding adjacent windows
  in both train and test. It does not guarantee separation by deployment/site/date.
- Classes missing training support are skipped and reported. Unsupported held-out
  metrics are unavailable rather than invented values.
- Bacpipe logs epoch losses. Saved metrics contain classification results, not
  an epoch-loss history.
- Saved weights include species ordering and embedding configuration metadata.
  Prediction checks dimensions and finite outputs.

See [tests/test_model.py](tests/test_model.py) for reload compatibility, splits,
configuration mismatches, and candidate coverage.

## Candidate scoring

Read [src/scoring.py](src/scoring.py) and [src/candidates.py](src/candidates.py).

Score every compatible window × trained species pair, including previously
reviewed windows. Save only the lowest-scoring N per species, default 100,
configured by `[candidates] candidates_per_species`. Break ties by window ID;
keep all available windows if fewer than N exist. `score_candidate(p, context)` returns `abs(2*p - 1)`:
zero at p=0.5, one at p=0 or p=1. Smaller scores receive review priority.
The context argument allows later factors such as existing annotations.

This is an uncalibrated heuristic based on multiclass softmax, not independent
presence confidence or overall multiclass entropy. An evenly spread multiclass
prediction need not put any individual species near 0.5.

Prediction uses bounded batches; selection retains at most N rows per species.
Write selected rows to a temporary JSONL file, then rename it after success.
Check per-species limits, stable ties, and lowest-score selection separately
from classifier accuracy. `iter_candidates` exposes exhaustive scores locally.

`publish <run-directory>` uploads only unique windows
referenced by that run's saved candidate file. Previously uploaded windows remain
available to existing annotations. Older saved runs are not retroactively limited.

## Cloud writes and failure recovery

Read [src/publish.py](src/publish.py), [src/cloud/storage.py](src/cloud/storage.py),
[src/cloud/bigquery.py](src/cloud/bigquery.py), and [tests/test_cloud.py](tests/test_cloud.py).

| Boundary | Recovery rule |
| --- | --- |
| Import | Invalid inputs fail by default; `--skip-invalid` explicitly permits reported exclusions. |
| Cache | Validate arrays and WAV headers; rebuild an affected recording when invalid. |
| Local mappings | Register after all recording artifacts are complete. |
| Snapshots | Write a temporary sibling directory, then rename; existing snapshots are immutable. |
| GCS | Create only; existing objects must match local size and checksum. |
| BigQuery windows | Merge checked metadata; mark local registration only after success. |
| Candidates | Stage, assert count/uniqueness/audio/species validity, then transactionally replace. |

There is no cross-system transaction across disk, GCS, and BigQuery. Interruption
can leave uploaded audio awaiting metadata registration; retries reconcile it.
A failed candidate-generation stage can leave saved model files without complete
candidate output. Directory existence alone does not prove a successful run.

`publish.json` records publication progress. A failed replacement transaction keeps
the previous candidate queue. Publication replaces candidates only, preserving
windows and annotator answers. Annotation reads join windows rather than candidates.

Read [structs/create_tables.sql](structs/create_tables.sql) for the schema.
[structs/reset_test_data_window_audio.sql](structs/reset_test_data_window_audio.sql)
is a destructive one-time reset, not routine setup.

Cloud clients use Application Default Credentials. The read-only access check
does not prove write permissions; actual operations enforce them.
Annotation refresh checks BigQuery only; publication checks BigQuery and GCS.
Local JSON reports/model metadata use [src/files.py](src/files.py), which writes
a temporary file before replacing the target and rejects unsupported values by default.

## Inspect one existing run

Review these outputs without regenerating the full dataset:

| Path under the artifact directory | What to inspect |
| --- | --- |
| `catalog.sqlite` | Identities, vector row mappings, registration state |
| `imports/<id>/import.json` | Generated/reused counts, errors, snapshot location |
| `imports/<id>/annotations/` | Selected CSV rows, exclusions, bootstrap snapshot |
| `annotations/<id>/` | Refreshed consensus, raw answers, unmatched IDs |
| `latest.json` | Defaults for the next training operation |
| `runs/<id>/model.json`, `species.json`, `model.pt` | Model identity, species order, weights |
| `runs/<id>/split.csv`, `metrics.json`, `skipped_species.json` | Training population and evaluation |
| `runs/<id>/annotations.json` | Snapshot selected for this run |
| `runs/<id>/candidates.json`, `candidates.jsonl` | Counts and scored pairs |
| `runs/<id>/publish.json` | Progress if publication was attempted |

Pick a window from `split.csv`, locate its vector and WAV through the catalog,
and compare model species order with its candidate rows.

## Local checks

From `C:\OsaAnurans`, using the application's existing environment:

```powershell
osa_anurans_app\.venv\Scripts\python.exe -m pytest osa_anurans_app/tests -q
osa_anurans_app\.venv\Scripts\ruff.exe check osa_anurans_app/src osa_anurans_app/tests
osa_anurans_app\.venv\Scripts\ruff.exe format --check osa_anurans_app/src osa_anurans_app/tests
```

These tests use temporary local data and fake cloud clients; they do not publish
to production. Pipeline tests use a tiny embedding backend. Audio tests also
exercise bacpipe preparation. [VALIDATION.md](VALIDATION.md) describes separate
real Perch and deployed-cloud checks and their limitations.

Prioritize identities, unknown-label handling, recording splits, and candidate
replacement safety before CLI formatting or logging details.
