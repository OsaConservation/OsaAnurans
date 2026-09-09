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
2. Searches GBIF Sound occurrences by scientificName first.
3. Falls back to the matched GBIF Backbone taxonKey when needed.
4. Downloads every audio URL exposed in the occurrence media field.
5. Converts each successful recording to 16-bit PCM WAV using FFmpeg.
6. Saves a metadata CSV linking each downloaded file to its GBIF occurrence,
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
import shutil
import subprocess
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


def paged_sound_search(
    session: requests.Session,
    search_params: dict,
    sleep_seconds: float = 0.25,
):
    """
    Yield every GBIF occurrence matching a Sound-media query.

    GBIF occurrence-search pages have a maximum page size of 300 records.
    """
    offset = 0

    while True:
        params = {
            **search_params,
            "mediaType": "Sound",
            "limit": PAGE_SIZE,
            "offset": offset,
        }

        data = get_json(
            session,
            f"{GBIF_API}/occurrence/search",
            params=params,
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


def count_sound_occurrences(
    session: requests.Session,
    search_params: dict,
) -> int:
    """Return GBIF's count for a Sound-media occurrence query."""
    params = {
        **search_params,
        "mediaType": "Sound",
        "limit": 0,
    }
    data = get_json(
        session,
        f"{GBIF_API}/occurrence/search",
        params=params,
    )
    return int(data.get("count", 0) or 0)


def find_sound_occurrences(
    session: requests.Session,
    scientific_name: str,
    taxon_key: int | None = None,
    sleep_seconds: float = 0.25,
):
    """
    Search GBIF for Sound records by scientific name first.

    If the exact scientific-name query returns no Sound occurrences, fall
    back to the matched GBIF Backbone taxonKey. This is useful for records
    whose indexed taxonomy uses an accepted name or synonym rather than the
    exact name in the spreadsheet.

    Yields:
        (occurrence, search_method)
    """
    name_params = {"scientificName": scientific_name}
    name_count = count_sound_occurrences(session, name_params)

    if name_count > 0:
        print(
            f"  Scientific-name search found {name_count} "
            f"Sound occurrence(s)."
        )
        for occurrence in paged_sound_search(
            session,
            name_params,
            sleep_seconds=sleep_seconds,
        ):
            yield occurrence, "scientificName"
        return

    print("  Scientific-name search found no Sound occurrences.")

    if not taxon_key:
        print("  No usable taxonKey is available for fallback.")
        return

    key_params = {"taxonKey": int(taxon_key)}
    key_count = count_sound_occurrences(session, key_params)

    if key_count == 0:
        print("  taxonKey fallback also found no Sound occurrences.")
        return

    print(
        f"  taxonKey fallback found {key_count} "
        f"Sound occurrence(s)."
    )

    for occurrence in paged_sound_search(
        session,
        key_params,
        sleep_seconds=sleep_seconds,
    ):
        yield occurrence, "taxonKey"


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



def find_ffmpeg(explicit_path: str | None = None) -> str:
    """Find the FFmpeg executable or raise a useful error."""
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return str(candidate)
        resolved = shutil.which(explicit_path)
        if resolved:
            return resolved
        raise RuntimeError(
            f"FFmpeg was not found at or under: {explicit_path}"
        )

    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved

    raise RuntimeError(
        "FFmpeg is required to convert recordings to WAV, but it was not "
        "found on PATH.\n"
        "Install FFmpeg first, then run this script again.\n"
        "Windows (winget): winget install Gyan.FFmpeg\n"
        "macOS (Homebrew): brew install ffmpeg\n"
        "Ubuntu/Debian: sudo apt install ffmpeg"
    )


def convert_to_wav(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
) -> tuple[bool, str]:
    """
    Convert an audio file to uncompressed 16-bit PCM WAV.

    The source sample rate and channel count are preserved unless FFmpeg
    itself needs to normalize them for WAV output.
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        return True, "already_exists"

    partial_wav = output_path.with_suffix(".wav.part")

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(partial_wav),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)

    if result.returncode != 0:
        if partial_wav.exists():
            try:
                partial_wav.unlink()
            except OSError:
                pass
        message = (result.stderr or "FFmpeg conversion failed").strip()
        return False, message

    try:
        partial_wav.replace(output_path)
    except OSError as exc:
        return False, str(exc)

    return True, "converted"

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
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="Optional path/name of the FFmpeg executable. "
             "By default the script searches PATH.",
    )
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Keep the publisher's original downloaded audio file after "
             "successful WAV conversion.",
    )
    args = parser.parse_args()

    try:
        ffmpeg = find_ffmpeg(args.ffmpeg)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Using FFmpeg: {ffmpeg}")

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
        "search_method",
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
        "wav_file",
        "download_status",
        "conversion_status",
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

            if taxon_key:
                print(
                    f"  GBIF match: {matched_name} "
                    f"(taxonKey={taxon_key}, "
                    f"matchType={match_type}, confidence={confidence})"
                )
            else:
                print(
                    "  No usable GBIF taxonKey match; "
                    "trying scientificName search anyway."
                )

            species_dir = audio_dir / sanitize_filename(requested_name)
            species_dir.mkdir(parents=True, exist_ok=True)

            found_media = 0
            downloaded_media = 0

            try:
                occurrences = find_sound_occurrences(
                    session,
                    requested_name,
                    taxon_key=taxon_key,
                    sleep_seconds=args.delay,
                )

                for occurrence, search_method in occurrences:
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

                        # The publisher's original format is downloaded to
                        # a temporary/original file, then converted to WAV.
                        extension = choose_extension(source_url, media)
                        original_path = species_dir / f"{base}{extension}"
                        wav_path = species_dir / f"{base}.wav"

                        print(
                            f"  Downloading {found_media}: "
                            f"GBIF {gbif_id} "
                            f"[{search_method}]"
                        )

                        # A completed WAV means this item is already usable;
                        # don't re-download it on a resumed run.
                        if wav_path.exists() and wav_path.stat().st_size > 0:
                            success = True
                            content_type = ""
                            download_status = "skipped_wav_exists"
                            conversion_ok = True
                            conversion_status = "already_exists"
                        else:
                            success, content_type = download_file(
                                session,
                                source_url,
                                original_path,
                            )

                            # If the URL/metadata did not reveal an extension,
                            # rename using the HTTP Content-Type when possible.
                            if success and extension == ".audio":
                                real_ext = choose_extension(
                                    source_url,
                                    media,
                                    content_type,
                                )
                                if real_ext != ".audio":
                                    new_original = original_path.with_suffix(
                                        real_ext
                                    )
                                    if not new_original.exists():
                                        original_path.replace(new_original)
                                        original_path = new_original

                            if success:
                                download_status = "downloaded"
                                print("    Converting to WAV...")
                                conversion_ok, conversion_message = convert_to_wav(
                                    ffmpeg,
                                    original_path,
                                    wav_path,
                                )
                                conversion_status = conversion_message
                            else:
                                download_status = "failed"
                                conversion_ok = False
                                conversion_status = "not_attempted"

                        if success and conversion_ok:
                            downloaded_media += 1
                            downloaded_file = (
                                str(original_path.relative_to(output_dir))
                                if original_path.exists()
                                else ""
                            )
                            wav_file = str(
                                wav_path.relative_to(output_dir)
                            )

                            if (
                                not args.keep_originals
                                and original_path.exists()
                                and original_path != wav_path
                            ):
                                try:
                                    original_path.unlink()
                                    downloaded_file = ""
                                except OSError as exc:
                                    print(
                                        f"    Could not delete original: {exc}"
                                    )
                        else:
                            downloaded_file = (
                                str(original_path.relative_to(output_dir))
                                if original_path.exists()
                                else ""
                            )
                            wav_file = ""
                            if success and not conversion_ok:
                                print(
                                    "    WAV conversion failed: "
                                    f"{conversion_status}"
                                )

                        writer.writerow(
                            {
                                "requested_species": requested_name,
                                "matched_species": matched_name,
                                "taxon_key": taxon_key,
                                "match_status": match_type,
                                "search_method": search_method,
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
                                "wav_file": wav_file,
                                "download_status": download_status,
                                "conversion_status": conversion_status,
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
