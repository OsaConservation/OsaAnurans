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

Defaults read new audio recursively under `osa_anurans_app/inputs/audio-inbox`.
Copy new recordings there; subdirectories are included. Labels come from BigQuery.
Species manifests and their reference clips also live under `inputs`.
Generated data, models, and logs live under `outputs`.
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
--embedding-config <id>` selects a particular saved dataset. Inference remains reserved;
adding species references is available in the opening checklist.

The opening screen shows the workflow checklist directly:

- Use Up/Down to move, Space to toggle steps, and Enter to continue.
- Continue and Exit are below the checklist.
- Select import, fetch annotations, train, upload candidates, and/or remove unused cloud audio.
- Fetch + train are selected by default; uploading and cleanup are unchecked.
- Uploading automatically includes training so it uses the new run's candidates.
- Cleanup runs after uploading when both are selected. It can also run on its own.
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

## Inputs, outputs, and indexing

```text
inputs/
  audio-inbox/                         # new recordings to ingest
  species.json                         # configured species manifest (local)
  species.example.json                 # example manifest to copy/edit
  references/                          # five-second species reference clips
outputs/
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
window has local audio. Cloud media URIs belong to published candidates, not windows.

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
- Target an approximately 80/20 recording split per species, with whole recordings
  on one side. Prefer selections with representative window counts too.
  A species with only one labeled recording stays entirely in training, with no
  test metrics. Shared recordings can limit balance; training coverage takes priority.
  Stable seeded ordering avoids reshuffling existing species when an unrelated
  new species is added. This does not freeze a permanent test set as labels grow.
  If no recording can be held out safely, train on all recordings and mark
  held-out evaluation unavailable.
- Select up to `candidates_per_species` candidates per species (default 100).
  Similarity receives `similarity_balance / (similarity_balance + positive_recordings)`
  of the allocation; the remaining candidates use classifier uncertainty.
  Configure both under `[candidates]`; `similarity_balance` defaults to 25.
  Count distinct recordings with present consensus, excluding held-out recordings.
- Search local embeddings by maximum cosine similarity to positive seeds.
  Exclude self-matching seeds from similarity results and held-out recordings
  from both pools. Multi-positive windows can be seeds. No classifier is required
  for similarity-only runs; species without positive seeds are skipped.
  Similarity also excludes window/species pairs with present or absent consensus.
  Unsure/tied answers remain eligible, and labels for other species do not exclude
  a window. Classifier uncertainty may revisit ordinary annotated pairs.
  Both methods exclude supplied reference/template seed pairs, identified by
  annotator IDs `new-species` and `eoin-template-seed` in the snapshot's original
  answers. These clips remain available for training and similarity seeds.
- Deduplicate, refill shortages from the other pool, and put similarity first.
  Save queue priority as `position / max(selected_count - 1, 1)` in `score`;
  Cloud Run serves lower scores first. Raw similarity/uncertainty and counts stay
  local in `candidate-selection.json`. Uploaded `selection_method` records
  `similarity` or `uncertainty`; raw measurements are not uploaded.
  Uncertainty remains `abs(2*p - 1)`, an uncalibrated multiclass heuristic.
- Store independent species annotations so later multilabel training can use
  explicit present/absent labels without changing window identities.

## Optional cloud workflow (later)

Local conversion never calls these actions automatically:

```powershell
gcloud auth application-default login
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli refresh
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli train
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli publish <run-directory>
# Optional: upload and then remove unused cloud clips.
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli publish <run-directory> --cleanup
# Cleanup only, including retrying an interrupted cleanup.
osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli cleanup
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
never uploaded. A bulk GCS listing detects missing clips even when SQLite marks
them registered; publication restores selected audio before replacing candidates.
Cloud adapters have fake-client tests and have passed a live migration integration
check covering uploads, metadata, replacement SQL, API pagination, and submission.
See VALIDATION.md; those temporary cloud fixtures were removed afterward.

`cleanup` removes only canonical `<window_id>.wav` objects under the configured
window prefix without a current candidate or reference. Annotation-only clips are removed.
It also finds GCS-only orphans left by interrupted uploads. References, local
WAVs/embeddings, and BigQuery recordings/windows/annotations are preserved.
Progress shows removed clips and bytes; `cloud-cleanup.json` records standalone
cleanup results, or `cleanup.json` beside the published run when run after upload.
After cleanup, the console also shows GCS active/soft-deleted storage and BigQuery
logical table storage in decimal GB. Totals are saved in the cleanup report.
BigQuery metadata can lag recent writes; this is stored table data, not query usage.

Cleanup uses the current candidate queue, with no grace period. Historical answers
train from local embeddings and do not retain cloud audio. An old browser batch may
lose media after queue replacement; its answers can still be submitted. Missing
current candidate clips are repaired when local audio is available. GCS soft-deleted bytes
remain until the bucket's retention period expires; cleanup does not change that policy.

Uploads and cleanup share a create-only GCS lock so they cannot overlap across
computers. Completed actions and handled errors release it. Ctrl+C, a process
crash, or shutdown leaves it in place because a submitted BigQuery job may still
be running. First verify no cloud operation or outstanding BigQuery job is running,
then remove `gs://<bucket>/<prefix>/.publication.lock` before retrying.
Retrying a run interrupted during cleanup preserves the current cloud queue.
Use `publish <run-directory> --cleanup` to resume its cleanup, or run `cleanup`
directly without training. Without that option, publish retries do not delete clips.

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
## Add species references

Select **Add species references and seed annotations** in the opening menu, then
read the configured JSON manifest. Review the names and cloud destinations before starting.
This step is unchecked by default. It writes seed/reference data to Google Cloud
even when candidate upload is unchecked; it preserves the existing candidate queue.

The menu and workflow review show the configured manifest path; no path entry is needed.
The default is `osa_anurans_app/inputs/species.json`; configure it with
`paths.species_manifest`. Use [inputs/species.example.json](inputs/species.example.json)
as the input format:

```json
{
  "species": [
    {
      "scientific_name": "Smilisca phaeota",
      "common_name": "New Granada cross-banded treefrog",
      "reference_audio": "references/smilisca-phaeota.mp3"
    }
  ]
}
```

- WAV or MP3, exactly five decoded seconds; finite, nonzero mono signal required.
- Relative audio paths resolve beside the manifest; absolute paths also work.
- Scientific names use two alphabetic words. IDs use the first three letters of
  each word, e.g. `Smilisca phaeota` → `SMIPHA`.
- Matching existing species may receive another reference or a missing seed.
  Conflicting IDs/names fail; existing metadata is never overwritten.
- Multiple references may repeat a species with matching names and distinct audio.
  Matching prepared audio for the same species is rejected before cloud uploads,
  even when source files use different encodings.
- Reviewed input changes stop the workflow. Source files are never edited.

The action saves local embeddings and exact model-input WAVs. Cloud audio goes to
`references/<species>/<reference>.wav`. Seed window clips stay local until selected
as candidates; adding a reference does not upload a redundant window copy.
Existing references are reused after comparing their audio; their IDs, URIs, and
spectrograms are preserved. When a reference has no spectrogram URL, the website
generates one from its audio using the same renderer as candidate spectrograms.

One stable `present` annotation per species/window is added with annotator
`new-species`. Rerunning the same inputs preserves votes and creation timestamps.
All metadata is committed in one transaction under the publication/cleanup lock.
Failures retain local files and uploaded objects for retry; progress/errors are
recorded in `outputs/species-imports/<attempt>/receipt.json`.

After committing, the action fetches the complete BigQuery annotation snapshot.
A separately selected refresh step reuses that result. Training stops if add or
refresh fails. Nonempty local CSV label snapshots must be seeded to BigQuery first.
The active corpus embedding configuration must match the reference backend.
Reference-only workspaces still need corpus audio to produce useful candidates.

```powershell
# Add only (explicit cloud write; no training or candidate publication).
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli add-species --manifest osa_anurans_app\inputs\species.json
```

Menu combinations:

- **Add species**: prepare references, upload metadata/seed votes, refresh labels.
- **Add species + Train**: also generate a new local candidate run.
- **Add species + Train + Upload**: also publish that new candidate run.


### Candidate media schema cutover

The new app requires media/source fields on candidates and metadata-only windows.
Do not run cloud writers against the old schema. Pause publication, cleanup,
add-species, and seed scripts until the cutover is complete:

1. Run `structs/migrate_candidate_media.sql` to snapshot tables and backfill candidates.
2. Check actual selected GCS objects, repairing missing clips before cleanup.
3. Deploy the updated API and install the updated app, keeping cloud writers paused.
4. Run `structs/drop_window_media.sql` after checking the backfill and API responses.
5. Resume writers; enable cleanup only after checking the current queue/media.

Saved local selection JSONL files remain supported. Publication creates a separate
`candidate-upload.jsonl` with verified audio URIs and copied source metadata.
Rollback after column removal requires restoring the window columns/data from the
snapshot before running old code. Keep cleanup disabled during rollback.
