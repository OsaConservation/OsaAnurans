from pydub import AudioSegment



import os
from pathlib import Path

def wav_to_mp3(directory="C:/OsaAnurans/docs/static/"):
    directory = Path(directory)
    for filename in directory.rglob("*.wav"):
        print(f"converting: {filename}")
        old_path = directory / filename
        new_path = old_path.with_suffix(".mp3")
        AudioSegment.from_wav(old_path).export(new_path, format="mp3")
        
        print(f"converted: {filename} -> {os.path.basename(new_path)}")

# Run the function in the current directory
wav_to_mp3()

            
