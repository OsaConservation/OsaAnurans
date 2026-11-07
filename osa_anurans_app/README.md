# Osa Anurans classifier workflow

Local-first conversion of recordings into Perch v2 embeddings, five-second audio
windows, annotation snapshots, a linear classifier, and review candidates.
See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the design.
Start with [CODE_REVIEW_GUIDE.md](CODE_REVIEW_GUIDE.md) for a reading order,
key design decisions, and checks to use while reviewing the implementation.

## Windows setup

Run from `C:\OsaAnurans`:

```powershell
uv sync --project osa_anurans_app --locked
```

The app uses its own `.venv` and Python 3.11. The complete dependency lock
includes bacpipe 1.3.5 and a Windows-compatible TensorFlow filesystem dependency.
The first embedding run downloads the approximately 414 MB Perch checkpoint.
The app automatically uses `config.local.toml` when it exists, otherwise
`config.example.toml`. Copy the example to `config.local.toml` to customize
paths/settings. Use `--config <path>` only to override this default.

Open the interactive menu without entering a config path:

```powershell
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli
```

Defaults read new audio recursively under `osa_anurans_app/audio-inbox`.
Copy new recordings there; subdirectories are included. Labels come from BigQuery.
Relative config paths resolve against the config directory.

## First local run

Import a small sample from the audio inbox:

```powershell
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli --config osa_anurans_app/config.example.toml import --limit-recordings 10
```

Import all new audio, then fetch labels and train:

```powershell
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli --config osa_anurans_app/config.example.toml import
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli --config osa_anurans_app/config.example.toml refresh
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli --config osa_anurans_app/config.example.toml train
```

Import and training are local; refresh reads BigQuery using local credentials.
Audio-only imports preserve saved labels and leave new windows unknown.
An empty inbox has nothing to import; existing data can still be used for training.
An optional `annotations_csv` setting supports one-time seed imports.
Run without an action for the interactive menu. Separate `import`, `train`, and
`status` actions are also available. `train --snapshot <directory>
--embedding-config <id>` selects a particular saved dataset. Inference and adding
species remain reserved menu actions.

The opening screen shows the workflow checklist directly:

- Use Up/Down to move, Space to toggle steps, and Enter to continue.
- Continue and Exit are below the checklist.
- Select import, fetch annotations, train, and/or upload candidates.
- Fetch + train are selected by default; uploading is unchecked.
- Uploading automatically includes training so it uses the new run's candidates.
- Review the plan and choose **Start workflow** or **Go back**. Esc cancels.
- After starting, all selected steps run without prompts. A failure stops later
  steps and keeps completed work.

Epochs and candidates per species stay in the config file. To train with saved
labels, uncheck fetching annotations. Uploading replaces the website's candidate queue and
preserves existing annotations. Direct `train` and `both` commands remain local.

The current Grillos inventory has 680 WAV files, about 24.8 hours of audio, one
zero-byte WAV, and one CSV reference to a missing H75 file. Imports stop on invalid
or missing input by default, before loading the model. To explicitly skip
unavailable audio and save an exclusion report, add `--skip-invalid` after
`import` or `both`. Input files are never modified. A full run produces about
17,900 windows; FLOAT WAV output alone needs about 11.5 GB.

`device = 'auto'` selects available acceleration and otherwise reports CPU.
Explicit `cuda` fails if the relevant backend cannot activate it. The default
locked ONNX Runtime package is CPU; GPU use requires compatible hardware and
ONNX Runtime CUDA/PyTorch installations and has not been validated on this machine.

## Output and indexing

```text
artifacts/
  catalog.sqlite                       # local identities and embedding row index
  checkpoints/perch_v2/
  embeddings/perch_v2/<config>/<id>.npy # one array per recording
  windows/v1/<window_id>.wav            # exact Perch input: 5s mono 32kHz FLOAT WAV
  imports/<id>/import.json              # completed/skipped/failed recordings
  imports/<id>/annotations/             # CSV, exclusions, consensus snapshot
  annotations/<id>/                    # optional BigQuery snapshots
  runs/<id>/                           # model, split, metrics, candidates.jsonl
  latest.json                          # active embedding config and snapshot
  pipeline.log                         # includes partial-overlap warnings
```

Recording IDs derive from file-content hashes; identical bytes share an identity
and path aliases. Re-encoding changes identity. Window IDs use recording ID,
source-sample bounds, and extraction policy. The last window uses bacpipe's wrap
padding across the recording. Recording offsets are metadata: window playback
starts at zero.

The SQLite `window_embeddings` table is local only and maps a window/config to a
`.npy` file and row. An embedding-model change does not itself change window ID.
An import refuses to overwrite an existing waveform with different samples:
preprocessing changes need an extraction-policy version change. Every generated
window has local audio; cloud synchronization later adds its required GCS URI.

Completed compatible artifacts are reused. Missing audio rebuilds its recording;
interrupted work is retryable. Do not run concurrent imports into one artifact
folder. Snapshots and classifier runs have unique directories.

## Training rules

- CSV columns: `audiofilename,start,end,label:species`. Any positive-duration
  overlap establishes presence; unexpected partial overlap is logged. Unlabeled
  regions remain unknown.
- Initially use single-label multiclass softmax training. Exactly one positive
  species per training window; exclude unknown, absent-only, and multi-positive
  windows. At least two species must have training examples.
- Use bacpipe's `LinearProbe` and `train_linear_probe` with our catalog-backed
  dataloader and recording-level split. Epoch loss is logged by bacpipe; run
  metrics store train/test classification results rather than loss history.
- Keep recordings wholly on one side of a seeded approximately 80/20 split.
  Keep every species represented in training, reducing the holdout if needed.
  If no recording can be held out safely, train on all recordings and mark
  held-out evaluation unavailable.
- Score every compatible window x trained species pair, including reviewed
  pairs, then save up to 100 lowest-scoring candidates per species.
  Configure this with `[candidates] candidates_per_species` in the TOML file.
  Ties are broken by window ID; species with fewer windows retain all of them.
  `score_candidate(p, context)` returns `abs(2*p - 1)`; smaller scores
  receive review priority. This is an uncalibrated heuristic, not an independent
  species-presence probability.
- Store independent species annotations so later multilabel training can use
  explicit present/absent labels without changing window identities.

## Optional cloud workflow (later)

Local conversion never calls these actions automatically:

```powershell
gcloud auth application-default login
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli refresh
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli train
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli publish <run-directory>
```

Confirm configured destinations before use. `publish <run-directory>` reads the saved
candidate file and uploads only its unique windows' pending audio to
`gs://osa-frog-data/windows/v1/<window_id>.wav`, verifies checksums, then inserts
missing recording/window metadata. Shared windows are uploaded once; unrelated
local windows remain local. It then replaces the cloud candidate queue.

`refresh` counts every present/absent annotator answer per window/species, ignores
unsure, and saves strict-majority consensus locally. Ties stay unknown. It neither
joins current candidates nor merges initial CSV labels back into the snapshot.
Unmatched window IDs are reported, including absent/unsure-only pairs.
Annotation refresh requires BigQuery access only; GCS access is checked for publication.

`publish` first uploads/registers only windows referenced by that saved run's
candidate file. It stages and validates candidates, then truncates/inserts them in one
BigQuery transaction. Existing annotations/windows/media remain; embeddings are
never uploaded. Previously registered GCS objects are not rechecked for later
external deletion on every publish. Cloud adapters have fake-client tests and have passed a live migration integration
check covering uploads, metadata, replacement SQL, API pagination, and submission.
See VALIDATION.md; those temporary cloud fixtures were removed afterward.

The limit applies when generating a new run. Older candidate files remain unchanged;
retrain to create a file with the configured limit before uploading it.

Cloud and local records use the same window IDs, with no ID translation.
The deployed API uses standalone window audio; the updated frontend source uses
zero-based playback/spectrogram bounds and keeps recording offsets as metadata.
Frontend changes still need a GitHub push before any website publication.

## Tests

```powershell
osa_anurans_app\.venv\Scripts\python.exe -m pytest osa_anurans_app/tests -q
osa_anurans_app\.venv\Scripts\ruff.exe check osa_anurans_app/src osa_anurans_app/tests
osa_anurans_app\.venv\Scripts\ruff.exe format --check osa_anurans_app/src osa_anurans_app/tests
```

Routine integration tests use real WAV files and an injected lightweight embedding
backend: no checkpoint download or cloud access. They cover exact row/audio
alignment, cache recovery, voting, grouped training, model reload, all candidate
pairs, and publication failures.
