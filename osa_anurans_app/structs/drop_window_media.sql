-- Phase 2, after backfill/object checks and service/app cutover.
-- Old console binaries cannot write windows after this step; keep them stopped.
ASSERT NOT EXISTS (
  SELECT 1 FROM `osaanurans.osafrogdataset.candidates` c
  LEFT JOIN `osaanurans.osafrogdataset.windows` w USING (window_id)
  WHERE w.window_id IS NULL OR c.recording_id IS DISTINCT FROM w.recording_id
     OR c.recording_start_seconds IS DISTINCT FROM w.start_seconds
     OR c.recording_end_seconds IS DISTINCT FROM w.end_seconds
     OR c.audio_gcs_uri IS NULL
) AS 'Candidate metadata must be complete before dropping window media';
ALTER TABLE `osaanurans.osafrogdataset.windows`
DROP COLUMN IF EXISTS audio_gcs_uri,
DROP COLUMN IF EXISTS spectrogram_gcs_uri;
