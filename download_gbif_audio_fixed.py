#!/usr/bin/env python3
"""
Download audio recordings from GBIF for every species in an Excel file.

Input Excel:
    specieslist.xlsx

Expected column:
    scientific_name

Example:
    scientific_name
    Craugastor ranoides
    Allobates talamancae

The script:
1. Reads the species names from Excel.
2. Matches each name against the GBIF Backbone Taxonomy.
3. Searches GBIF occurrence records with mediaType=Sound.
4. Downloads every audio URL exposed in the occurrence media field.
5. Saves a metadata CSV linking each downloaded file to its GBIF occurrence,
   species, dataset, license, rights holder, and source URL.

GBIF occurrence searches are paginated at up to 300 records per request.
The script deliberately downloads sequentially to be polite to GBIF and
publisher servers.

GBIF API documentation:
https://techdocs.gbif.org/en/openapi/v1/occurrence
https://techdocs.gbif.org/en/openapi/v1/species
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import mimetypes
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


GBIF_API = "https://api.gbif.org/v1"
PAGE_SIZE = 300

# A descriptive User-Agent is useful when accessing public biodiversity APIs.
USER_AGENT = (
    "GBIF-Audio-Downloader/1.0 "
    "(research script; https://www.gbif.org/)"
)

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac",
    ".aiff", ".aif", ".wma", ".opus", ".webm", ".mp4"
}


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Make text safe for use in a filename."""
    text = re.sub(r"[^\w.\-]+", "_", str(text), flags=re.UNICODE)
    text = text.strip("._")
    return text[:max_length] or "unknown"


def get_json(session: requests.Session, url: str, params=None, retries: int = 5):
    """GET JSON with retry/backoff for temporary errors and rate limiting."""
    for attempt in range(retries):
        try:
            response = session.get(
                url,
                params=params,
                timeout=60,
            )

            if response.status_code == 429:
                wait = min(60, 5 * (2 ** attempt))
                print(f"  Rate limited (429); waiting {wait}s...")
                time.sleep(wait)
                continue

            if 500 <= response.status_code < 600:
                wait = min(60, 2 * (2 ** attempt))
                print(
                    f"  GBIF server error {response.status_code}; "
                    f"waiting {wait}s..."
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = min(60, 2 * (2 ** attempt))
            print(f"  Request failed: {exc}; waiting {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Unable to retrieve {url}")


def match_species(session: requests.Session, scientific_name: str) -> dict:
    """Match a name to the legacy GBIF Backbone integer taxonKey.

    The occurrence search endpoint expects the integer Backbone taxonKey.
    We therefore use the v1 matcher first (it returns ``usageKey``), then
    fall back to v2 with several response formats handled.
    """
    v1 = get_json(
        session,
        f"{GBIF_API}/species/match",
        params={"name": scientific_name},
    )

    taxon_key = v1.get("usageKey") or v1.get("taxonKey")
    if taxon_key:
        return {
            "taxonKey": taxon_key,
            "scientificName": v1.get("canonicalName") or v1.get("scientificName") or scientific_name,
            "matchType": v1.get("matchType", ""),
            "confidence": v1.get("confidence", ""),
            "status": v1.get("status", ""),
        }

    v2 = get_json(
        session,
        "https://api.gbif.org/v2/species/match",
        params={"scientificName": scientific_name},
    )
    usage = v2.get("usage") or {}
    taxon_key = (
        usage.get("key") or usage.get("usageKey") or usage.get("taxonKey")
        or v2.get("usageKey") or v2.get("taxonKey")
        or usage.get("acceptedUsageKey") or v2.get("acceptedUsageKey")
    )
    return {
        "taxonKey": taxon_key,
        "scientificName": usage.get("canonicalName") or usage.get("name") or v2.get("canonicalName") or v2.get("scientificName") or scientific_name,
        "matchType": v2.get("matchType", ""),
        "confidence": v2.get("confidence", ""),
        "status": usage.get("status") or v2.get("status", ""),
    }


def find_sound_occurrences(
    session: requests.Session,
    taxon_key: int,
    sleep_seconds: float = 0.25,
):
    """
    Yield occurrence records containing Sound media for one taxon.

    GBIF allows at most 300 records per occurrence-search page.
    """
    offset = 0

    while True:
        data = get_json(
            session,
            f"{GBIF_API}/occurrence/search",
            params={
                "taxonKey": taxon_key,
                "mediaType": "Sound",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
        )

        results = data.get("results", [])
        if not results:
            break

        for occurrence in results:
            yield occurrence

        total = data.get("count", 0)
        offset += len(results)

        if offset >= total or len(results) < PAGE_SIZE:
            break

        time.sleep(sleep_seconds)


def get_audio_media(occurrence: dict) -> list[dict]:
    """Return only media objects that GBIF identifies as Sound."""
    audio = []

    for media in occurrence.get("media", []) or []:
        media_type = str(media.get("type", "")).lower()
        identifier = media.get("identifier")

        if identifier and media_type == "sound":
            audio.append(media)

    return audio


def choose_extension(url: str, media: dict, content_type: str = "") -> str:
    """Determine a reasonable audio file extension."""
    # Prefer a known extension in the publisher URL.
    path_suffix = Path(urlparse(url).path).suffix.lower()
    if path_suffix in AUDIO_EXTENSIONS:
        return path_suffix

    # Then use the media format supplied by the publisher.
    fmt = str(media.get("format", "")).lower()
    if "/" in fmt:
        ext = mimetypes.guess_extension(fmt.split(";")[0].strip())
        if ext:
            return ext

    # Finally use the HTTP Content-Type.
    if "/" in content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext

    return ".audio"


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    retries: int = 4,
) -> tuple[bool, str]:
    """
    Download a single file.

    Returns:
        (success, content_type)
    """
    if destination.exists() and destination.stat().st_size > 0:
        return True, "already_downloaded"

    partial = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(retries):
        try:
            with session.get(
                url,
                stream=True,
                timeout=(30, 180),
                allow_redirects=True,
            ) as response:
                if response.status_code == 429:
                    wait = min(60, 5 * (2 ** attempt))
                    print(f"    429 from publisher; waiting {wait}s...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")

                with open(partial, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            fh.write(chunk)

                # Only rename after a complete successful response.
                partial.replace(destination)

                return True, content_type

        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"    Download failed: {exc}")
                break

            wait = min(30, 2 * (2 ** attempt))
            print(f"    Download error; waiting {wait}s...")
            time.sleep(wait)

        except OSError as exc:
            print(f"    File error: {exc}")
            break

    if partial.exists():
        try:
            partial.unlink()
        except OSError:
            pass

    return False, ""


def load_species(excel_path: Path, column: str) -> list[str]:
    """Read and clean species names from Excel."""
    df = pd.read_excel(excel_path)

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' was not found. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )

    names = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    names = [name for name in names if name]
    # Preserve input order while removing duplicate names.
    return list(dict.fromkeys(names))


def main():
    parser = argparse.ArgumentParser(
        description="Download GBIF Sound media for species listed in an Excel file."
    )
    parser.add_argument(
        "excel",
        nargs="?",
        default="specieslist.xlsx",
        help="Excel file containing species names (default: specieslist.xlsx)",
    )
    parser.add_argument(
        "--column",
        default="scientific_name",
        help="Excel column containing scientific names "
             "(default: scientific_name)",
    )
    parser.add_argument(
        "--output",
        default="gbif_audio",
        help="Output directory (default: gbif_audio)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Delay between GBIF search pages in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--limit-per-species",
        type=int,
        default=0,
        help="Optional maximum number of audio files per species; "
             "0 means no limit.",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "gbif_audio_metadata.csv"

    species = load_species(excel_path, args.column)

    print(f"Loaded {len(species)} unique species from {excel_path}")
    print(f"Audio output: {audio_dir.resolve()}")
    print()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Metadata is written incrementally so an interrupted run still leaves
    # a useful record of what happened.
    fieldnames = [
        "requested_species",
        "matched_species",
        "taxon_key",
        "match_status",
        "gbif_id",
        "occurrence_id",
        "dataset_name",
        "dataset_key",
        "publisher",
        "recorded_by",
        "event_date",
        "country",
        "license",
        "rights_holder",
        "media_type",
        "media_format",
        "media_created",
        "source_url",
        "downloaded_file",
        "download_status",
    ]

    # If a previous metadata file exists, append to it. This makes it possible
    # to resume a large run. Existing downloaded files are also skipped.
    metadata_exists = metadata_path.exists()

    with open(
        metadata_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as metadata_file:

        writer = csv.DictWriter(
            metadata_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        if not metadata_exists:
            writer.writeheader()

        for species_index, requested_name in enumerate(species, start=1):
            print(
                f"[{species_index}/{len(species)}] "
                f"{requested_name}"
            )

            try:
                match = match_species(session, requested_name)
            except Exception as exc:
                print(f"  ERROR matching species: {exc}")
                continue

            taxon_key = match.get("taxonKey")
            matched_name = match.get("scientificName", requested_name)
            match_type = match.get("matchType", "")
            confidence = match.get("confidence", "")

            if not taxon_key:
                print("  No GBIF taxon match.")
                continue

            print(
                f"  GBIF match: {matched_name} "
                f"(taxonKey={taxon_key}, "
                f"matchType={match_type}, confidence={confidence})"
            )

            species_dir = audio_dir / sanitize_filename(requested_name)
            species_dir.mkdir(parents=True, exist_ok=True)

            found_media = 0
            downloaded_media = 0

            try:
                occurrences = find_sound_occurrences(
                    session,
                    int(taxon_key),
                    sleep_seconds=args.delay,
                )

                for occurrence in occurrences:
                    gbif_id = occurrence.get("key")
                    occurrence_id = occurrence.get("occurrenceID", "")
                    media_items = get_audio_media(occurrence)

                    for media_index, media in enumerate(media_items, start=1):
                        if (
                            args.limit_per_species > 0
                            and found_media >= args.limit_per_species
                        ):
                            break

                        found_media += 1

                        source_url = media["identifier"]

                        # Hash the URL so multiple occurrences with identical
                        # or very long URLs get safe, deterministic filenames.
                        url_hash = hashlib.sha1(
                            source_url.encode("utf-8")
                        ).hexdigest()[:10]

                        base = (
                            f"{sanitize_filename(requested_name)}_"
                            f"gbif_{gbif_id}_"
                            f"{media_index}_{url_hash}"
                        )

                        # Start with an extension inferred from the URL or
                        # metadata. If unknown, .audio is used temporarily.
                        extension = choose_extension(source_url, media)
                        destination = species_dir / f"{base}{extension}"

                        print(
                            f"  Downloading {found_media}: "
                            f"GBIF {gbif_id}"
                        )

                        success, content_type = download_file(
                            session,
                            source_url,
                            destination,
                        )

                        # If the URL/metadata did not reveal an extension,
                        # rename the file using the HTTP content type.
                        if success and extension == ".audio":
                            real_ext = choose_extension(
                                source_url,
                                media,
                                content_type,
                            )
                            if real_ext != ".audio":
                                new_destination = destination.with_suffix(
                                    real_ext
                                )
                                if not new_destination.exists():
                                    destination.replace(new_destination)
                                    destination = new_destination

                        if success:
                            downloaded_media += 1
                            status = "downloaded"
                            downloaded_file = str(
                                destination.relative_to(output_dir)
                            )
                        else:
                            status = "failed"
                            downloaded_file = ""

                        writer.writerow(
                            {
                                "requested_species": requested_name,
                                "matched_species": matched_name,
                                "taxon_key": taxon_key,
                                "match_status": match_type,
                                "gbif_id": gbif_id,
                                "occurrence_id": occurrence_id,
                                "dataset_name": occurrence.get(
                                    "datasetName", ""
                                ),
                                "dataset_key": occurrence.get(
                                    "datasetKey", ""
                                ),
                                "publisher": occurrence.get(
                                    "publishingOrgKey", ""
                                ),
                                "recorded_by": occurrence.get(
                                    "recordedBy", ""
                                ),
                                "event_date": occurrence.get(
                                    "eventDate", ""
                                ),
                                "country": occurrence.get(
                                    "country", ""
                                ),
                                "license": occurrence.get(
                                    "license", ""
                                ),
                                "rights_holder": occurrence.get(
                                    "rightsHolder", ""
                                ),
                                "media_type": media.get("type", ""),
                                "media_format": media.get("format", ""),
                                "media_created": media.get(
                                    "created", ""
                                ),
                                "source_url": source_url,
                                "downloaded_file": downloaded_file,
                                "download_status": status,
                            }
                        )
                        metadata_file.flush()

                    if (
                        args.limit_per_species > 0
                        and found_media >= args.limit_per_species
                    ):
                        break

            except Exception as exc:
                print(f"  ERROR searching occurrences: {exc}")
                continue

            print(
                f"  Found {found_media} audio files; "
                f"downloaded {downloaded_media}."
            )

    print()
    print("Finished.")
    print(f"Audio files: {audio_dir.resolve()}")
    print(f"Metadata:    {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
