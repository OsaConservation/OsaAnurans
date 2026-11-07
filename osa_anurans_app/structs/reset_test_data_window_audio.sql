-- Destructive one-time test reset, explicitly approved by the project owner.
-- Run in project osaanurans, location us-south1.
-- species and reference_media MUST remain untouched.
CREATE OR REPLACE TABLE `osaanurans.osafrogdataset.deployments` (
  deployment_id STRING NOT NULL,
  name STRING NOT NULL,
  project STRING NOT NULL,
  latitude FLOAT64,
  longitude FLOAT64
);

CREATE OR REPLACE TABLE `osaanurans.osafrogdataset.recordings` (
  recording_id STRING NOT NULL,
  deployment_id STRING,
  recorded_at TIMESTAMP
);

CREATE OR REPLACE TABLE `osaanurans.osafrogdataset.windows` (
  window_id STRING NOT NULL,
  recording_id STRING NOT NULL,
  start_seconds FLOAT64 NOT NULL,
  end_seconds FLOAT64 NOT NULL,
  audio_gcs_uri STRING NOT NULL,
  spectrogram_gcs_uri STRING
);

CREATE OR REPLACE TABLE `osaanurans.osafrogdataset.embeddings` (
  window_id STRING NOT NULL,
  model_config JSON NOT NULL,
  embedding ARRAY<FLOAT64>
);

CREATE OR REPLACE TABLE `osaanurans.osafrogdataset.candidates` (
  candidate_id STRING NOT NULL,
  window_id STRING NOT NULL,
  query_species_id STRING NOT NULL,
  created_at TIMESTAMP NOT NULL,
  score FLOAT64 NOT NULL
);

CREATE OR REPLACE TABLE `osaanurans.osafrogdataset.annotations` (
  annotation_id STRING NOT NULL,
  window_id STRING NOT NULL,
  candidate_id STRING,
  query_species_id STRING NOT NULL,
  response STRING NOT NULL,
  annotator_id STRING NOT NULL,
  annotator_name STRING NOT NULL,
  submission_id STRING NOT NULL,
  created_at TIMESTAMP NOT NULL
);
