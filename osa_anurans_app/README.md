# Console app guide

Import audio, fetch labels, train a classifier, and publish candidates for annotation.
For the website and its local server, start with the [main README](../README.md).

## Start the app

Run from `C:\OsaAnurans` on Windows. Install Python 3.11, uv, and the Google Cloud
CLI first.

```powershell
# Install the locked dependencies.
uv sync --project osa_anurans_app --locked

# Sign in for cloud access. You do not need to repeat this for every run.
gcloud auth application-default login

# Open the menu.
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli
```

- You need access to the project's BigQuery dataset and GCS bucket for cloud steps.
- The first embedding run downloads about 414 MB of Perch model weights.
- The default installation supports CPU. CUDA needs a compatible GPU setup.
- Settings come from `config.local.toml` when present, otherwise `config.example.toml`.

## Choose a workflow

- **Import new audio:** read recordings from `inputs/audio-inbox/`; save five-second clips and embeddings locally.
- **Import labeled audio:** read `inputs/species.json`; add labels and optional references to Google Cloud.
- **Fetch latest annotations:** save BigQuery votes as a local label snapshot.
- **Train classifier:** save the model and train/test results locally. Do not generate candidates.
- **Generate candidates:** use current labels and the new or latest compatible saved classifier; save a separate selection locally. Without a classifier, use similarity only.
- **Upload candidates:** upload selected clips and replace the website's candidate queue. Existing annotations stay.
- **Remove unused cloud audio clips:** delete window clips without current candidates or references. Keep local files and BigQuery metadata/annotations.

Use **Up/Down** to move, **Space** to toggle, and **Continue** to review your choices.
Choose **Start workflow** to run everything without more prompts. **Esc** cancels.

- Fetch + train + generate are selected by default.
- Uploading automatically includes generation, but does not force training.
- For evaluation only, select **Train classifier** and uncheck generation/upload.
- To reuse a saved model, select **Generate candidates** without training.
- To train from saved labels, uncheck fetching.
- A failed step stops the workflow; completed work is kept for retry.

### Example: import recordings and update the website

1. Copy recordings into `osa_anurans_app/inputs/audio-inbox/`. Subfolders work too.
2. Open the app.
3. Select **Import new audio**, **Fetch latest annotations**, **Train**, **Generate candidates**, and **Upload**.
4. Review the plan and start.
5. Open the website to label the new candidates.

New recordings start with unknown labels. Importing audio does not assume a species
is absent. Completed imports are reused on later runs.

## Browse saved annotations

Choose **Browse annotations** in the opening menu, or run:

```powershell
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli browse
```

- Open the printed local URL. Stop the server with **Ctrl+C**.
- Each row is one saved answer, including repeated annotations of the same clip.
- Filter by species, answer, or annotator name; search recording filenames/window IDs.
- Select a row to listen and generate its spectrogram. Changing pages clears playback.
- Source offsets describe the original recording; each saved clip plays from zero.
- Missing local audio stays visible as an unavailable clip.
- Browsing uses local files and needs no cloud credentials or model. It does not edit answers.
- Run `refresh` to fetch newer answers, then restart `browse`. Reloading the page alone
  does not fetch annotations. Snapshots containing only aggregate CSV labels need a refresh
  before individual answers can be shown.
- Use `browse --snapshot <directory>` for an older snapshot, or `browse --port 8123`
  if the default port 8765 is occupied. Selecting an older snapshot leaves training defaults unchanged.

## Import labeled clips or add a species

Edit `osa_anurans_app/inputs/species.json`, then choose **Import labeled audio**.

- Use WAV or MP3 clips that are **exactly five seconds** long.
- Audio paths are relative to the JSON file; absolute paths also work.
- Use a species ID for an existing species, or scientific/common names to declare one.
- Use `present` or `absent`. Labels for other species remain unknown.
- Set `add_as_reference` to `true` to show a present clip in the website's reference carousel.
- A new species needs at least one present reference in the batch.

### Example: label clips for an existing species

```json
{
  "species": [
    {
      "species_id": "SCIBOU",
      "audio": "labeled-audio/scibou-present.wav",
      "response": "present"
    },
    {
      "species_id": "SCIBOU",
      "audio": "labeled-audio/scibou-absent.wav",
      "response": "absent"
    }
  ]
}
```

### Example: add a species with a reference clip

```json
{
  "species": [
    {
      "scientific_name": "Smilisca phaeota",
      "common_name": "New Granada cross-banded treefrog",
      "audio": "references/smilisca-phaeota.mp3",
      "response": "present",
      "add_as_reference": true
    }
  ]
}
```

- IDs use the first three letters of each scientific-name word: `Smilisca phaeota` becomes `SMIPHA`.
- Matching species are reused; conflicting names/IDs are rejected.
- Embeddings and window clips are saved locally. Only reference clips are uploaded during this step.
- Labels are written to BigQuery, then all annotations are fetched. A separate fetch step reuses that result.
- This step writes to the cloud **even when candidate upload is unchecked**; it keeps the current queue.
- References use annotator ID `new-species`; label-only clips use `labeled-audio-import`.
- Reimporting identical files adds no votes. Moving a file is fine; re-encoding changes its identity.
- Editing an imported answer or changing its reference status is not supported yet.

## Settings and files

To customize settings, copy `config.example.toml` to `config.local.toml` if you do
not already have one. Paths in that file are relative to its directory.

```toml
[paths]
audio_roots = ['inputs/audio-inbox']
species_manifest = 'inputs/species.json'
artifacts = 'outputs'

[training]
epochs = 50
device = 'auto'

[candidates]
candidates_per_species = 100
similarity_balance = 25
```

- Edit the existing sections in your config; keep its cloud settings.
- Epochs and candidate limits are config settings, not menu options.
- Change the output directory only when you intend to use a different local dataset.

```text
osa_anurans_app/
  inputs/
    audio-inbox/          New recordings
    species.json          Labeled-audio manifest
    references/           Reference clips you supply
    labeled-audio/        Other labeled clips you supply
  outputs/
    catalog.sqlite        Recording/window IDs and embedding locations
    checkpoints/          Cached Perch model
    embeddings/           Local vectors
    windows/v1/           Five-second audio clips
    annotations/          Saved BigQuery votes and consensus
    runs/                 Separate runs for models/results and candidate selections
    imports/              Audio import reports
    species-imports/      Labeled-audio import receipts
    pipeline.log          Progress and errors
```

- Keep `outputs/`: training depends on its local embeddings.
- Embeddings are never uploaded to Google Cloud.
- Identical recording bytes share an ID. Windows use that ID and their source bounds.
- Short final windows repeat audio to reach five seconds; standalone playback starts at zero.
- Run one import at a time per output directory.

## How labels and candidates work

- **Consensus:** count present vs. absent votes; ignore unsure. A strict majority wins; ties stay unknown. Repeated submissions count as additional votes.
- **Training:** use windows with exactly one present species. Unknown, absent-only, and multiple-present windows are excluded.
- **Test split:** keep whole recordings together, targeting roughly 80% training / 20% test per species. Single-recording species stay in training and have no held-out metrics.
- **Similarity:** find clips close to positive examples in embedding space.
- **Uncertainty:** find clips the classifier is less certain about. This is a heuristic, not calibrated confidence.
- **Mix:** fewer positive recordings means more similarity candidates. The default similarity fraction is `25 / (25 + positive_recordings)`.
- **Exclusions:** similarity skips clips already labeled present or absent for that species. Uncertainty may revisit ordinary labels. Both skip test recordings and supplied reference/template clips for that species.
- **Small datasets:** if there are too few classes to train, report that no classifier was trained. Candidate generation can still use similarity only.
- **Serving:** saved scores represent queue rank. The API samples with a preference for lower scores; separate website sessions can receive different orders.

## Individual commands

Use these instead of the menu when you want one step. All examples run from the
repository root.

```powershell
# Import a small sample locally.
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli import --limit-recordings 10

# Fetch votes, then train locally.
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli refresh
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli train

# Generate candidates separately, reusing the latest compatible classifier.
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli generate

# Import labeled audio and optional references.
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli add-species --manifest osa_anurans_app/inputs/species.json

# Replace RUN_ID with a directory created by candidate generation.
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli publish osa_anurans_app/outputs/runs/RUN_ID

# Remove unused cloud window clips.
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli cleanup
```

- `train` never publishes by itself.
- `generate` never trains. Use `generate --model-directory path/to/model-run` to choose a particular saved classifier.
- Generation reads the current labels and preserves the saved model's original test-recording exclusions. It creates a new selection without overwriting earlier runs.
- Add `--cleanup` to `publish` to clean up after uploading.
- Add `--skip-invalid` to `import` to skip bad audio and save an exclusion report.
- Global options go before the command: `...src.cli --config path/to/config.toml train`.

## Cloud behavior and recovery

- Publication uploads only selected candidate clips, verifies them, and inserts missing recording/window metadata before replacing the queue.
- Cleanup protects current candidates and references. Annotation-only cloud clips can be deleted because training uses local embeddings.
- An old website batch can lose its audio after queue replacement and cleanup; its answers can still be submitted.
- Cleanup reports removed bytes plus current GCS and BigQuery storage. Soft-deleted GCS bytes remain until retention expires.
- Cloud writers share a GCS lock. After an interrupted run, confirm no app or BigQuery job is still writing before removing a stale `.publication.lock` object.
- Do not rerun publication just to retry cleanup; run `cleanup` directly.
- Current code expects candidate media/source fields. Reference ordering also needs the `display_order` column. Migration SQL is in [structs/](structs/); production has already been migrated.

## Run checks

```powershell
.\osa_anurans_app\.venv\Scripts\python.exe -m pytest osa_anurans_app/tests -q
.\osa_anurans_app\.venv\Scripts\python.exe -m ruff check osa_anurans_app/src osa_anurans_app/tests
```

- Tests use small test embeddings and fake cloud clients; they do not write production data.
- Source starts in [src/cli.py](src/cli.py), [src/pipeline.py](src/pipeline.py), and [src/species.py](src/species.py).
