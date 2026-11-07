"""Add-only species metadata batches; never replace candidates or existing rows."""

from uuid import uuid4

from .bigquery import _bigquery

TABLE_KEYS = {
    "species": "species_id",
    "reference_media": "reference_id",
    "recordings": "recording_id",
    "windows": "window_id",
    "annotations": "annotation_id",
}


def read_catalog(bq) -> tuple[list[dict], list[dict]]:
    """Check reference schema/access only for this action, then read catalog rows."""
    fields = {field.name for field in bq.client.get_table(f"{bq.dataset}.reference_media").schema}
    required = {"reference_id", "species_id", "audio_gcs_uri", "spectrogram_gcs_uri"}
    if not required <= fields:
        raise ValueError(f"reference_media is missing fields: {sorted(required - fields)}")
    species = [dict(row) for row in bq._query(f"SELECT * FROM {bq._table('species')}")]
    references = [dict(row) for row in bq._query(f"SELECT * FROM {bq._table('reference_media')}")]
    return species, references


def insert_batch(bq, tables: dict[str, list[dict]]) -> None:
    """Stage all five tables and insert missing keys in one transaction.

    - Compare every supplied field, including nulls; preserve original timestamps.
    - Recheck name collisions and duplicate target keys inside the transaction.
    - Failed assertions roll back the entire metadata batch.
    """
    api = _bigquery()
    stages = {}
    statements = ["BEGIN TRANSACTION;"]
    try:
        for table, key in TABLE_KEYS.items():
            rows = tables[table]
            if not rows:
                raise ValueError(f"Empty species batch table: {table}")
            if len({row[key] for row in rows}) != len(rows):
                raise ValueError(f"Duplicate staged {table} keys")
            columns = list(rows[0])
            schema = {
                field.name: field for field in bq.client.get_table(f"{bq.dataset}.{table}").schema
            }
            fields = [api.SchemaField(name, schema[name].field_type) for name in columns]
            stage = f"_species_{table}_{uuid4().hex}"
            # Track before loading so failed loads also get cleaned up.
            stages[table] = stage
            bq._load_rows(rows, stage, fields)
            target, source = bq._table(table), bq._table(stage)
            comparisons = [
                f"t.{name} IS DISTINCT FROM s.{name}"
                for name in columns
                if name != key and name != "created_at"
            ]
            statements.append(f"""
                ASSERT NOT EXISTS (
                    SELECT t.{key} FROM {target} t JOIN {source} s USING ({key})
                    GROUP BY t.{key} HAVING COUNT(*) > 1
                ) AS 'Duplicate existing {table} keys';
                ASSERT NOT EXISTS (
                    SELECT 1 FROM {target} t JOIN {source} s USING ({key})
                    WHERE {" OR ".join(comparisons) if comparisons else "FALSE"}
                ) AS 'Conflicting existing {table} metadata';
            """)
            if table == "species":
                statements.append(f"""
                    ASSERT NOT EXISTS (
                        SELECT 1 FROM {target} t JOIN {source} s
                        ON LOWER(REGEXP_REPLACE(TRIM(t.scientific_name), r'\\s+', ' '))
                         = LOWER(s.scientific_name)
                        WHERE t.species_id != s.species_id
                    ) AS 'Scientific name already has another species ID';
                """)
            statements.append(f"""
                MERGE {target} t USING {source} s ON t.{key} = s.{key}
                WHEN NOT MATCHED THEN INSERT ({", ".join(columns)})
                VALUES ({", ".join("s." + name for name in columns)});
            """)
        statements.append("COMMIT TRANSACTION;")
        bq._query("\n".join(statements))
    finally:
        for stage in stages.values():
            bq.client.delete_table(f"{bq.dataset}.{stage}", not_found_ok=True)
