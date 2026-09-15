# BirdNET 2.4 embedding extractor

Extracts one 1024-D BirdNET 2.4 acoustic embedding per 3-second WAV segment.

## Windows command

```powershell
$env:BIRDNET_APP_DATA="C:\BirdNETData"
python extract_birdnet_embeddings.py `
  --metadata "D:\Acoustics\AnuraSet_3sec\INCT4\metadata.csv" `
  --audio-root "D:\Acoustics\AnuraSet_3sec\INCT4" `
  --output-dir "D:\Acoustics\AnuraSet_3sec\INCT4\birdnet_embeddings"
```

Resume an interrupted run with `--resume`.

## Outputs

- `embeddings.npy`: N x 1024 float32 matrix, same row order as metadata
- `extraction_status.csv`
- `metadata_with_embedding_status.csv`
- `failed_segments.csv`
- `extraction_summary.json`

The extractor uses `result.embeddings[0, 0, :]`, mono-converts stereo audio,
and uses one worker/batch item at a time for stability on Windows.

`weak_recording_labels` are preserved as recording-level metadata and are not
turned into segment-level labels.
