# One-time Grillos seed

Run from `C:\OsaAnurans`, using a dedicated seed config with `annotations_csv` set.
The completed seed used `artifacts/grillos-seed-config.toml`; the normal
configs now point to the audio inbox and must not be used for seeding.
Use the same seed config for every command below:

```powershell
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli --config osa_anurans_app/artifacts/grillos-seed-config.toml import --skip-invalid
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.scripts.seed_grillos --config osa_anurans_app/artifacts/grillos-seed-config.toml
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli --config osa_anurans_app/artifacts/grillos-seed-config.toml refresh
.\osa_anurans_app\.venv\Scripts\python.exe -m osa_anurans_app.src.cli --config osa_anurans_app/artifacts/grillos-seed-config.toml train
```

- Import generates all local WAVs, embeddings, and CSV presence labels.
- Seed registers annotated windows and recording metadata without uploading audio.
- Seed inserts one present vote per window/species under `grillos-seed-gt`.
- Duplicate CSV intervals do not become extra votes. Multiple species are retained.
- Metadata and annotation IDs are reusable; retrying does not add votes.
- Seed does not replace candidates or delete existing data.
- Refresh reads BigQuery votes; train reads that snapshot and local embeddings.
- After refresh, training needs neither this script nor the original Grillos folder.
- Keep the catalog, embedding arrays, and local WAVs for future training/publication.
- To retry seed after refresh, first rerun import to select the CSV snapshot again.

Publishing the resulting candidates remains an explicit, separate CLI command.
