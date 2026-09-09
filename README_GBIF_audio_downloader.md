# GBIF audio downloader

This script reads `specieslist.xlsx`, matches each scientific name to the GBIF
Backbone Taxonomy, finds GBIF occurrence records whose media type is `Sound`,
and downloads the audio files exposed by those records.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

Put these three files in the same directory:

- `download_gbif_audio.py`
- `specieslist.xlsx`
- `requirements.txt`

Then:

```bash
python download_gbif_audio.py
```

The default Excel column is `scientific_name`.

Output:

```text
gbif_audio/
├── audio/
│   ├── Craugastor_ranoides/
│   ├── Allobates_talamancae/
│   └── ...
└── gbif_audio_metadata.csv
```

The metadata CSV records the GBIF occurrence ID, taxon key, source URL,
dataset, license, rights holder, and downloaded filename.

### Other options

Use a different Excel column:

```bash
python download_gbif_audio.py specieslist.xlsx --column scientific_name
```

Use a different output folder:

```bash
python download_gbif_audio.py --output my_recordings
```

Limit downloads during testing:

```bash
python download_gbif_audio.py --limit-per-species 10
```

Increase the delay between GBIF search pages:

```bash
python download_gbif_audio.py --delay 1.0
```

## Important

GBIF indexes media supplied by publishers; GBIF does not necessarily host the
audio itself. The `source_url` in the metadata CSV is the publisher URL used
for the download. Respect the license and any publisher-specific terms for
each recording.

For very large collections, GBIF recommends using its asynchronous occurrence
download service rather than making a very large number of occurrence-search
requests.
