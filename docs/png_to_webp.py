from PIL import Image

from pathlib import Path



import os

def png_to_webp(directory=r"C:\OsaAnurans\docs\img"):
    directory = Path(directory)
    for filename in list(directory.rglob("*.png")):
        old_path = directory / filename
        new_path = old_path.with_suffix(".webp")

        with Image.open(old_path) as img:
            img.save(new_path, "WEBP", quality = 80)
            print(f"converted: {filename} -> {os.path.basename(new_path)}")

# Run the function in the current directory
png_to_webp()

            
