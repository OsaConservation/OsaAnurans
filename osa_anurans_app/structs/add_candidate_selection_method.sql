-- Preserve the existing queue; new publications populate this field.
ALTER TABLE `osaanurans.osafrogdataset.candidates`
ADD COLUMN IF NOT EXISTS selection_method STRING;
