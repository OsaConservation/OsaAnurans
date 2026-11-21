-- Apply before deploying the reference-ordering API. Existing order is preserved.
ALTER TABLE `osaanurans.osafrogdataset.reference_media`
ADD COLUMN IF NOT EXISTS display_order INT64;
