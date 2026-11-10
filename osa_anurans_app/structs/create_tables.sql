-- Run in BigQuery Studio, project osaanurans, processing location us-south1.
-- Existing tables/data are preserved. IF NOT EXISTS does not migrate schemas.
CREATE TABLE IF NOT EXISTS `osaanurans.osafrogdataset.species` (
  species_id STRING NOT NULL,
  scientific_name STRING NOT NULL,
  common_name STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `osaanurans.osafrogdataset.reference_media` (
  reference_id STRING NOT NULL,
  species_id STRING NOT NULL,
  audio_gcs_uri STRING NOT NULL,
  spectrogram_gcs_uri STRING,
  -- Lower values appear first; NULL uses the normal reference order (1).
  display_order INT64
);

CREATE TABLE IF NOT EXISTS `osaanurans.osafrogdataset.deployments` (
  deployment_id STRING NOT NULL,
  name STRING NOT NULL,
  project STRING NOT NULL,
  latitude FLOAT64,
  longitude FLOAT64
);

CREATE TABLE IF NOT EXISTS `osaanurans.osafrogdataset.recordings` (
  recording_id STRING NOT NULL,
  deployment_id STRING,
  recorded_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `osaanurans.osafrogdataset.windows` (
  window_id STRING NOT NULL,
  recording_id STRING NOT NULL,
  start_seconds FLOAT64 NOT NULL,
  end_seconds FLOAT64 NOT NULL
);

CREATE TABLE IF NOT EXISTS `osaanurans.osafrogdataset.embeddings` (
  window_id STRING NOT NULL,
  model_config JSON NOT NULL,
  embedding ARRAY<FLOAT64>
);

CREATE TABLE IF NOT EXISTS `osaanurans.osafrogdataset.candidates` (
  candidate_id STRING NOT NULL,
  window_id STRING NOT NULL,
  query_species_id STRING NOT NULL,
  created_at TIMESTAMP NOT NULL,
  score FLOAT64 NOT NULL,
  selection_method STRING,
  recording_id STRING NOT NULL,
  recording_start_seconds FLOAT64 NOT NULL,
  recording_end_seconds FLOAT64 NOT NULL,
  audio_gcs_uri STRING NOT NULL,
  spectrogram_gcs_uri STRING
);

CREATE TABLE IF NOT EXISTS `osaanurans.osafrogdataset.annotations` (
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
