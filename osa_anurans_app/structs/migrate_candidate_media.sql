-- Phase 1: pause all console cloud writers first. Do not enable cleanup during cutover.
-- Retain snapshots for rollback; this script never clears data.
CREATE SNAPSHOT TABLE IF NOT EXISTS `osaanurans.osafrogdataset.windows_before_candidate_media`
CLONE `osaanurans.osafrogdataset.windows`;
CREATE SNAPSHOT TABLE IF NOT EXISTS `osaanurans.osafrogdataset.candidates_before_candidate_media`
CLONE `osaanurans.osafrogdataset.candidates`;

ALTER TABLE `osaanurans.osafrogdataset.candidates`
ADD COLUMN IF NOT EXISTS recording_id STRING,
ADD COLUMN IF NOT EXISTS recording_start_seconds FLOAT64,
ADD COLUMN IF NOT EXISTS recording_end_seconds FLOAT64,
ADD COLUMN IF NOT EXISTS audio_gcs_uri STRING,
ADD COLUMN IF NOT EXISTS spectrogram_gcs_uri STRING;

BEGIN TRANSACTION;
ASSERT NOT EXISTS (
  SELECT window_id FROM `osaanurans.osafrogdataset.windows`
  GROUP BY window_id HAVING COUNT(*) > 1
) AS 'Duplicate window IDs';
UPDATE `osaanurans.osafrogdataset.candidates` c
SET recording_id = w.recording_id,
    recording_start_seconds = w.start_seconds,
    recording_end_seconds = w.end_seconds,
    audio_gcs_uri = w.audio_gcs_uri,
    spectrogram_gcs_uri = w.spectrogram_gcs_uri
FROM `osaanurans.osafrogdataset.windows` w
WHERE c.window_id = w.window_id;
ASSERT NOT EXISTS (
  SELECT 1 FROM `osaanurans.osafrogdataset.candidates`
  WHERE recording_id IS NULL OR recording_start_seconds IS NULL
     OR recording_end_seconds IS NULL OR recording_start_seconds < 0
     OR IS_NAN(recording_start_seconds) OR IS_INF(recording_start_seconds)
     OR IS_NAN(recording_end_seconds) OR IS_INF(recording_end_seconds)
     OR ABS(recording_end_seconds - recording_start_seconds - 5) > 0.000001
     OR audio_gcs_uri IS NULL
     OR audio_gcs_uri != CONCAT('gs://osa-frog-data/windows/v1/', window_id, '.wav')
) AS 'Incomplete or invalid candidate media backfill';
COMMIT TRANSACTION;
-- Before phase 2: check actual selected GCS objects, repair missing clips, deploy
-- the new service, and install the new app. Keep all console cloud writes paused.
