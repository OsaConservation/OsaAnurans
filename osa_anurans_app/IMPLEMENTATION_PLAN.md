# Annotation and classifier pipeline implementation plan

Status: implemented with a local-first initial workflow. See README.md for executable commands and validation limits. Optional cloud operations are separate commands and have not been run against production.

## 1. Agreed behavior

- Use bacpipe and Perch v2 to generate embeddings. Windows are exactly 5 seconds. Verify bacpipe's default stride and final-window behavior before implementation; record the effective settings with every embedding run.
- Read recordings recursively from `C:\Users\DELL EMC\Desktop\Grillos`, including its SWIFT subfolder. All initial annotations come from `C:\Users\DELL EMC\Desktop\Grillos\annotations.csv`; do not import the SWIFT selection table separately.
- Local interval labels establish presence only. Unlabeled species/window pairs remain unknown, including when another species is present.
- Aggregate BigQuery answers by `(window_id, query_species_id)`: count every present/absent answer, ignore unsure, use strict majority. Ties and zero votes are unknown. Do not deduplicate annotators or select their latest answer.
- Initially train a single-label multiclass linear probe: one species label per training window. This explicitly assumes the selected labeled species is the sole target class for that example, matching the notebook. Preserve per-window/per-species annotations for a later independent multilabel classifier. Unlabeled windows remain excluded from training.
- Split by recording, approximately 80% training and 20% testing, using a reproducible seed.
- Initial multiclass mode needs positive examples for at least two species in the training partition; it does not require explicit absent examples per species. Skip species without positive training examples and warn. The later multilabel mode will require both present and absent examples per trained species.
- Keep embeddings local. Save the classifier, label mapping, configuration, and metrics locally.
- Generate candidates for every local window and every successfully trained species, including previously annotated pairs. Initial multiclass candidate scoring is specified below; do not treat absolute raw multiclass logits as uncertainty.
- Replace the candidates table. Preserve recordings, windows, media, and annotations. Upload only missing windows/media associated with newly processed recordings; reuse existing windows on later runs.
- Leave the menu actions for standalone inference and adding species unimplemented, with an explanatory message.

## 2. CLI and configuration

Initial operation is local only: `both` imports Desktop/Grillos, writes local
five-second WAVs/embeddings/annotation snapshots, trains, and writes local candidate
JSONL. No cloud credentials or writes occur. Explicit `refresh` and
`publish` commands expose later cloud integration. Audio uploads use a saved candidate file to select windows. Until upload, local
windows have local paths and pending state rather than fabricated GCS URIs. See README.md for exact output layout.

The implementation uses bacpipe 1.3.5's Perch Embedder/ONNX backend and verified
5-second nonoverlapping windows with wrap padding. It uses a bacpipe-compatible
linear head with a small explicit training loop to guarantee the recording split
and avoid cached probing-dataframe alignment. `uv.lock` pins the tested environment.


Entry point: `python -m osa_anurans_app.src.cli --config <path>` from the repository root.

The interactive menu offers:

1. Generate embeddings from recordings.
2. Retrain classifier and replace candidates.
3. Run 1, then 2.
4. Run inference — reserved.
5. Add new species — reserved.

Before processing, display resolved input/output paths, configured cloud destinations, model, and window settings. For training, ask whether to refresh the BigQuery annotation snapshot or use the local snapshot. The initial CSV is the bootstrap annotation input. Refreshing from BigQuery writes a consensus snapshot; using locally saved annotations reads that snapshot. Do not automatically merge the initial CSV back into a refreshed snapshot. Missing snapshots fail with an actionable message rather than silently training without them.

`config.toml` supplies source roots and label formats, output directories, project/dataset/bucket, GCS prefix, device selection, random seed, and training settings. Credentials use Google Application Default Credentials; passwords and access tokens are not stored in config. Ship `config.example.toml` and ignore machine-specific config and generated outputs.

Suggested output layout:

```text
artifacts/
  catalog.sqlite
  embeddings/perch_v2/<model-config-hash>/<recording-id>.npy
  annotations/<snapshot-id>/answers.csv
  annotations/<snapshot-id>/consensus.csv
  annotations/<snapshot-id>/snapshot.json
  runs/<run-id>/config.json
  runs/<run-id>/split.csv
  runs/<run-id>/model.pt
  runs/<run-id>/species.json
  runs/<run-id>/metrics.json
  runs/<run-id>/skipped_species.json
  runs/<run-id>/publish.json
  windows/v1/<window-id>.wav
  checkpoints/perch_v2/
```

The local catalog is the source of alignment between recordings, windows, and embedding rows. Never rely on directory enumeration order or a flattened embedding array's implicit order.

## 3. Key data structures

Use typed dataclasses for application records and NumPy arrays for numerical data. Pandas is limited to CSV/dataframe interoperability where useful.

| Structure | Key fields and purpose |
| --- | --- |
| `AppConfig` | Source configurations, paths, cloud destinations, model/device, seed, training settings. |
| `SourceConfig` | Stable source name, recordings root, annotation path/format, optional deployment metadata. |
| `RecordingRecord` | Stable ID, source-relative path, content hash, sample rate, frame count, optional deployment/time. Detect changed files; avoid basename collisions. |
| `WindowRecord` | Stable ID, recording ID, exact bounds in original-recording samples, source sample rate, window policy/padding, audio URI and persistent upload state. No embedding-model fields. |
| `WindowEmbeddingRecord` | Local-only mapping from `(window_id, embedding_config_id)` to embedding artifact and row index. Never uploaded to BigQuery. |
| `EmbeddingArtifact` | Recording ID, array path, shape/dtype, model identity/configuration, window metadata, completion state. |
| `IntervalLabel` | Recording ID, species ID, start/end seconds, source/provenance. Presence only. |
| `LabelCounts` | Window/species, present votes, absent votes; derived consensus is present, absent, or unknown. |
| `TrainingDataset` | Initially embeddings `X[N,D]`, integer class targets `y[N]`, window IDs, recording IDs, ordered species IDs. Keep the underlying per-species consensus snapshot so a future multilabel dataset can add targets `Y[N,S]` and observed-label mask `M[N,S]`. |
| `DatasetSplit` | Train/test row indices and recording IDs, seed, label support counts. |
| `ProbeArtifact` | Model parameters, ordered species IDs, embedding compatibility metadata, training settings, run ID. |
| `CandidateRecord` | Candidate ID, window ID, query species ID, UTC creation time, score; exactly matches BigQuery fields. |
| `PublishManifest` | Run ID, staged table, required windows/media, upload progress, candidate counts, publication state. Supports recovery. |

Proposed IDs: deterministic UUIDs derived from a versioned namespace and recording content identity, then window sample bounds. Candidate IDs include the training run, window, and species. Existing BigQuery window IDs must be reused through an explicit catalog mapping; do not regenerate identities for existing windows.

Keep embeddings in `.npy` files with memory-mapped reads, not in SQLite or BigQuery. SQLite stores searchable metadata and processing/upload state. Commit catalog completion only after atomically writing a complete embedding file and its window mapping.

## 4. Files and interfaces

The existing `src/model` files are empty and can retain their intended responsibilities. Add small modules around them rather than putting orchestration in model code.

| File | Main responsibilities and proposed functions |
| --- | --- |
| `src/cli.py` | `main()`, `choose_action()`, `display_paths()`, `choose_annotation_source()`; input and concise progress summaries. |
| `src/config.py` | `load_config(path) -> AppConfig`, `validate_config()`; resolve relative paths against config location. |
| `src/records.py` | Shared dataclasses and enums. |
| `src/pipeline.py` | `generate_embeddings(config)`, `train_and_publish(config, refresh_annotations)`, `run_both(config)`; sequence stages and propagate failures. |
| `src/catalog.py` | SQLite migrations; recording/window lookup, artifact registration, pending uploads, stable identity mapping. |
| `src/sources.py` | `discover_recordings()`, `read_interval_labels()`, `resolve_species()`; read the single initial CSV and resolve recording paths across configured roots without ambiguous basename matches. |
| `src/annotations.py` | `aggregate_votes()`, `map_intervals_to_windows()`, snapshot export/load with provenance. |
| `src/bacpipe_adapter.py` | Isolate version-specific bacpipe APIs: `BacpipePerch.prepare_audio()` and `embed_frames()` with pinned/checksummed ONNX weights. The training boundary lives in `src/model`; return explicit window-to-row mapping. |
| `src/model/dataloader.py` | Batch/memory-mapped embedding reads and initial single-label batches. |
| `src/model/datasets.py` | `build_dataset()` and `split_by_recording()`; align IDs, single-label targets, and recording groups. |
| `src/model/model.py` | Linear species probe interface, raw logits, serialization, embedding compatibility checks. |
| `src/model/train.py` | `train_classifier()`; train on windows with exactly one positive species; return artifact and skipped-species report. |
| `src/model/evaluation.py` | `evaluate_classifier()`; per-species metrics/support using only observed test labels. |
| `src/model/validate.py` | Dataset/model preconditions, split isolation, finite values, shape/identity checks. |
| `src/scoring.py` | `score_candidate(probability, context) -> float`; initially `abs(2*p - 1)`. Context can later include annotation counts without changing publication code. |
| `src/candidates.py` | `iter_candidates()`; batched logits and streamed records for all windows × trained species. |
| `src/cloud/bigquery.py` | Species/annotation reads, existing window lookup, idempotent metadata merge, staged candidate load and replacement. |
| `src/cloud/storage.py` | Upload missing window audio, verify objects, deterministic object names; no full-recording uploads. |
| `src/audio.py` | Extract exactly the model's 5-second window, including matching any padding policy; retain canonical local FLOAT WAV audio for reruns and later synchronization. |
| `src/publish.py` | `prepare_publish()`, `upload_missing_windows()`, `replace_candidates()`; recoverable publication sequence. |

Use package `__init__.py` files as needed. Keep cloud clients injectable for tests. UI prompts stay in the CLI; library functions accept arguments and return results.

## 5. Embedding generation and window identity

1. Discover audio recursively within each configured source and report file counts.
2. Identify recordings by content hash and source metadata. Detect duplicate content without confusing different files with the same basename.
3. Skip compatible, completed embedding artifacts. A model/version/window configuration change creates a separate artifact configuration.
4. Use bacpipe's Perch v2 preprocessing and windowing. Capture actual window bounds and embedding row indices from the adapter.
5. Validate that each window is exactly 5 seconds of model input and each embedding row has one catalog window. Persist the final-window padding policy explicitly.
6. Write artifacts atomically, register completed windows, and leave interrupted work retryable.
7. Extract each window's standalone audio locally. After training, `publish <run-directory>` uploads only windows referenced by that run's candidate file and inserts missing recording/window metadata before replacing candidates in BigQuery. Persist pending uploads for retries. Never mark a cloud window ready without its required verified audio URI.

Progress includes recordings completed/skipped/failed, windows produced, elapsed time, and output paths. Do not change the process working directory into bacpipe's installation as the notebooks do.

## 6. Labels, splitting, and training

### Labels

Parse the single initial CSV's columns `audiofilename,start,end,label:species`. Validate species IDs against the species catalog. Match audio using configured roots and source-relative paths, with ambiguity treated as an error. No separate SWIFT annotation parser is needed.

Any positive-duration overlap marks a window present; touching endpoints alone does not. Partial overlap is unexpected: log a warning with the recording, species, interval bounds, window bounds, and overlap duration whenever a labeled interval overlaps but does not fully cover a window. Outside all intervals is unknown.

BigQuery snapshots retain raw answers and derived vote counts, plus fetch time and schema/version metadata. Use the strict-majority consensus described above. Preserve enough provenance to explain every training target.

The initial CSV supplies bootstrap presence labels. A BigQuery refresh produces one consensus result per window/species pair and saves it locally. Loading the cached snapshot uses those same results without querying again. There is no additional local-versus-cloud voting or automatic bootstrap-label merge. Ties and pairs without qualifying votes remain unknown.

### Split and model

Assign each recording to one side of a deterministic 80/20 split; apply it consistently across species. Report actual sizes and per-species class support. Do not move individual windows between sides to improve balance.

Initial mode uses a multiclass linear probe trained with cross-entropy on windows with exactly one positive species label. Exclude windows with no positive labels or multiple positive species; report exclusion counts. An absent-only label does not establish another species as present. Other species provide competing classes under the explicitly accepted single-label assumption; do not persist invented absent annotations.

Verify the pinned bacpipe implementation supports an explicit recording-disjoint split, linear multiclass training, and raw logits/probabilities. Reuse its probe facilities where possible, with a minimal adapter for missing hooks. Never silently use a random window-level split.

Require at least two species with positive training examples; otherwise stop before candidate replacement and explain why. Warn about species lacking training examples and about insufficient held-out support. Retain independent annotator votes and multiple positive species in stored snapshots even though the initial model excludes multi-positive windows.

Future multilabel mode will use independent logits, masked binary cross-entropy, and explicit present/absent consensus; unknown pairs will be masked and species missing either training class skipped.

Save class support, confusion matrix, accuracy, and per-class precision/recall/F1 using the predicted top class. These evaluate single-label discrimination, not independent species-presence detection. Save loss, split, seed, model settings, label ordering, and dependency versions. Do not use the test partition for early stopping or hyperparameter selection.

## 7. Candidate generation and cloud publication

Run batched inference over every compatible catalog window, not only labeled windows. Keep up to `candidates_per_species` lowest-scoring candidates per species (default 100), breaking ties by window ID. Upload only unique windows referenced by the saved selection. BigQuery score is computed exclusively through `score_candidate()`.

Initial multiclass score: for species probability `p`, use `abs(2*p - 1)` so lower scores prioritize probabilities near 0.5. Multiclass logits have an arbitrary common offset, so `abs(raw_logit)` is not a meaningful uncertainty measure here. This score is a review-priority heuristic, not calibrated confidence in independent presence. This heuristic is accepted for the initial uncalibrated model. Score all window/species pairs locally and retain only the lowest N per species; softmax probabilities sum to one and cannot represent simultaneous species presence. Later binary/multilabel mode can use the originally proposed `abs(binary_logit)`.

Publication order:

1. Produce a run manifest and load all candidates into a temporary BigQuery staging table.
2. Validate finite scores, IDs, uniqueness, and the expected `window_count × trained_species_count` rows.
3. Query existing windows. Upload missing window audio for newly processed recordings, verify successful objects, then merge recording/window metadata without modifying existing rows. Reuse a single audio object when multiple species share a window. Report an error for an old window whose required cloud media is missing rather than silently inventing metadata.
4. Verify every staged candidate references a persisted window with usable audio.
5. In a BigQuery transaction, truncate candidates and insert the validated staging rows. This preserves the requested replacement behavior while avoiding an empty or partial table when uploads fail.
6. Mark publication successful and clean up temporary resources. Retain recovery information on failure.

Do not delete historical windows, annotations, or media. Publishing candidates changes candidate IDs, but historical annotations remain valid through their window/species fields; training queries must not require a join to the current candidates table.

Embeddings are not uploaded. Spectrogram URI remains optional; the frontend can generate it. Deployment of the updated window-audio API and frontend compatibility is a separate integration checkpoint before using these candidates on the site. Window playback must start at zero within the standalone clip rather than applying recording offsets again.

Media destination: `gs://osa-frog-data/windows/v1/<window_id>.wav`. The bucket and prefix are user-specified. Stable window IDs allow reuse across classifier runs and species. Keep model/run IDs out of these object paths. The `v1` prefix versions the audio extraction format/policy; a future incompatible policy must also produce distinct window identities. Match the stored audio to the model's mono sample rate and padding policy.

## 8. Dependencies and environment

- Python 3.11 for the verified Windows dependency lock (bacpipe's pinned TensorFlow dependency restricts the supported interpreter).
- Target Windows with GPU preferred when the verified Perch backend supports it. Verify the actual backend's GPU support; do not assume that PyTorch CUDA availability means Perch can use the GPU. Report the selected device and any CPU fallback clearly.
- `bacpipe`: pin a verified version; the playground requests `>=1.3.5,<2.0.0`. It is not installed in the currently active environment.
- NumPy, pandas, PyTorch, soundfile, scikit-learn, tqdm: use compatible versions from the verified bacpipe environment and declare direct imports explicitly.
- `google-cloud-bigquery`, `google-cloud-storage`, and BigQuery dataframe support if used.
- Standard-library `dataclasses`, `sqlite3`, `tomllib`, `logging`, `uuid`, and hashing utilities.
- `pytest` for tests. No clustering, UMAP, plotting-dashboard, or custom AVEX dependencies are needed for this workflow.

Add an app-level `pyproject.toml` with a reproducible dependency lock. Verify Perch v2 runtime/device support on the intended OS before downloading models or running long jobs. Avoid copying the playground's entire dependency list without validation.

## 9. Tests and acceptance checks

- Label fixtures: all votes count, unsure ignored, ties unknown, presence intervals never imply absence, multiple species allowed, partial-overlap warnings and consensus snapshot reuse.
- Alignment fixtures: repeated basenames, renamed/changed recordings, deterministic IDs, exact embedding-row mapping, 5-second windows, final-window policy, safe reruns.
- Training fixtures: recording-disjoint split, unlabeled/multi-positive windows excluded, at least two training classes required, missing training classes skipped, test-only class shortages reported, artifact reload preserves logits/species order.
- Candidate fixtures: full Cartesian coverage, previously annotated pairs included, mode-specific score and replaceable scoring function, nonfinite values rejected.
- Publication fault injection: interrupted media upload, failed metadata merge, failed staging load, failed final transaction. Existing candidates survive every failure before commit; reruns reuse completed uploads.
- Small real Perch/bacpipe smoke run: a few recordings, known labels, saved/reloaded model, then staged cloud verification. Real cloud writes use a test dataset/bucket before production publication.
- Initial positive-only datasets with at least two supported species can train a multiclass model without explicit negatives. A one-class-only training dataset stops with a warning and preserves existing candidates.

## 10. Implementation sequence

1. Verify bacpipe interfaces/runtime and settle the review questions below.
2. Add configuration, typed records, catalog, source parsers, and annotation snapshots.
3. Implement embedding adapter and explicit window alignment; test on a few recordings.
4. Implement consensus, single-label dataset, recording split, probe training/evaluation, and artifact reload.
5. Implement standalone scoring and streamed candidate generation.
6. Implement recoverable audio/window uploads and transactional candidate replacement.
7. Connect CLI actions, documentation, and end-to-end checks.

## 11. Confirmed decisions and remaining verification

- Media prefix: `gs://osa-frog-data/windows/v1/`.
- Count any positive-duration interval overlap and warn on unexpected partial overlaps.
- BigQuery refresh writes consensus labels locally; cached mode reads them. No extra source-merging policy.
- Windows is the target platform, with GPU preferred. All initial annotations are in the single Grillos CSV.
- Verify bacpipe's window alignment, single-label training, recording split hooks, raw logits, and Windows GPU backend before implementation.
- Embedding mappings remain local. Cloud and local records share window IDs; report annotations whose windows are absent from the local catalog.
- Standalone-window frontend playback/spectrogram verification will be handled by the user later; it remains a deployment checkpoint.
- Bootstrap decision: start with single-label multiclass classification using positive examples across species; move to independent multilabel classification once sufficient explicit positive/negative annotator labels exist. Use the species-specific probability score above.
