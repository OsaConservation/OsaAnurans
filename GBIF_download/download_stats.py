import pandas as pd

df = pd.read_csv("D:/Acoustics/gbif_audio/gbif_audio_metadata.csv")

# ---------------------------------------------------------
# DOWNLOAD SUMMARY BY SPECIES
# ---------------------------------------------------------

# Adjust these column names if your downloader uses different names.
# Expected:
#   name       = species name
#   occurrence = GBIF occurrence that was found
#   downloaded = whether the audio was successfully downloaded
df['download_success'] = (df["download_success"] != "failed").astype(int) 
#This overestimates the amount of downloads, as wav files that already existed are counted twice. might be able to add && df["download_success"] != "already_exists" to the above line to fix this, but then it will underestimate the amount of downloads, as some files that were already downloaded might have been deleted and re-downloaded.
summary = (
    df
    .groupby("matched_species")
    .agg(
        sound_occurrences_found=("occurrence_id", "count"),
        successfully_downloaded=("download_success", "sum")
    )
    .reset_index()
)

# Calculate failed downloads
summary["failed_downloads"] = (
    summary["sound_occurrences_found"]
    - summary["successfully_downloaded"]
)

# Print summary
print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))

# Save summary
summary.to_csv("gbif_download_summary.csv", index=False)

print("\nSummary saved to: gbif_download_summary.csv")