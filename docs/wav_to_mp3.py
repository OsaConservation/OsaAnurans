from pydub import AudioSegment



import os

def wav_to_mp3(directory=r"C:\OsaAnurans\docs\audio"):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, filename[:-4] + ".mp3")
            AudioSegment.from_wav(old_path).export(new_path, format="mp3")
            
            print(f"converted: {filename} -> {os.path.basename(new_path)}")

# Run the function in the current directory
wav_to_mp3()

            
