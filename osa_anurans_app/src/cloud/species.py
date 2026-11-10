"""Add-only species metadata batches; never replace candidates or existing rows."""

from uuid import uuid4

from ..annotations import REFERENCE_SEED_ANNOTATORS
from .bigquery import _bigquery

TABLE_KEYS = {
    "species": "species_id",
    "reference_media": "reference_id",
    "recordings": "recording_id",
    "windows": "window_id",
    "annotations": "annotation_id",
}

LABEL_ANNOTATOR = "labeled-audio-import"


def check_import_conflicts(bq, annotations: list[dict]) -> None:
    """Reject changed imported labels and reference promotion before uploading audio."""
    parameter = _bigquery().ArrayQueryParameter(
        "window_ids", "STRING", sorted({row["window_id"] for row in annotations})
    )
    existing = bq._query(
        f"SELECT * FROM {bq._table('annotations')} WHERE window_id IN UNNEST(@window_ids)",
        [parameter],
    )
    desired = {(row["window_id"], row["query_species_id"]): row for row in annotations}
    for old in existing:
        row = desired.get((old["window_id"], old["query_species_id"]))
        if row is None:
            continue
        old_annotator, new_annotator = old.get("annotator_id"), row["annotator_id"]
        cross_provenance = (
            old_annotator in REFERENCE_SEED_ANNOTATORS and new_annotator == LABEL_ANNOTATOR
        ) or (old_annotator == LABEL_ANNOTATOR and new_annotator in REFERENCE_SEED_ANNOTATORS)
        if cross_provenance:
            raise ValueError(
                f"{row['query_species_id']}/{row['window_id']} already has an imported "
                "label or reference seed; changing its reference status is not supported"
            )
        if (
            old.get("annotation_id") == row["annotation_id"]
            and old.get("response") != row["response"]
        ):
            raise ValueError(
                "Conflicting existing imported response; label corrections are not supported"
            )


def _import_conflict_assertion(target: str, source: str) -> str:
    """Repeat the import-provenance guard inside the atomic metadata transaction."""
    reference_ids = ", ".join(f"'{value}'" for value in sorted(REFERENCE_SEED_ANNOTATORS))
    return f"""
        ASSERT NOT EXISTS (
            SELECT 1 FROM {target} t JOIN {source} s
            ON t.window_id = s.window_id AND t.query_species_id = s.query_species_id
            WHERE (t.annotator_id IN ({reference_ids})
                   AND s.annotator_id = '{LABEL_ANNOTATOR}')
               OR (t.annotator_id = '{LABEL_ANNOTATOR}'
                   AND s.annotator_id IN ({reference_ids}))
        ) AS 'Imported label/reference promotion is not supported';
    """


def read_catalog(bq, *, include_references: bool = True) -> tuple[list[dict], list[dict]]:
    """Read identities; skip reference access for label-only imports."""
    species = [dict(row) for row in bq._query(f"SELECT * FROM {bq._table('species')}")]
    if not include_references:
        return species, []
    fields = {field.name for field in bq.client.get_table(f"{bq.dataset}.reference_media").schema}
    required = {"reference_id", "species_id", "audio_gcs_uri", "spectrogram_gcs_uri"}
    if not required <= fields:
        raise ValueError(f"reference_media is missing fields: {sorted(required - fields)}")
    references = [dict(row) for row in bq._query(f"SELECT * FROM {bq._table('reference_media')}")]
    return species, references


def insert_batch(bq, tables: dict[str, list[dict]]) -> None:
    """Stage metadata and insert missing keys in one transaction.

    - Compare every supplied field, including nulls; preserve original timestamps.
    - Recheck name collisions and duplicate target keys inside the transaction.
    - Failed assertions roll back the entire metadata batch.
    - Reference rows are optional; label-only imports skip that table.
    """
    api = _bigquery()
    stages = {}
    statements = ["BEGIN TRANSACTION;"]
    try:
        for table, key in TABLE_KEYS.items():
            rows = tables[table]
            if not rows:
                if table == "reference_media":
                    continue
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
            if table == "annotations":
                statements.append(_import_conflict_assertion(target, source))
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
