import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
import sys
import os

def upsample_wav(input_path, output_path, target_rate):
    try:
        # Read audio file
        data, samplerate = sf.read(input_path)
        
        if samplerate == target_rate:
            print(f"Sampling rate is already {target_rate} Hz. No change made.")
            sf.write(output_path, data, samplerate)
            return
        
        # Calculate upsample/downsample factors
        # Example: from 22050 Hz to 44100 Hz → up=2, down=1
        gcd = np.gcd(samplerate, target_rate)
        up = target_rate // gcd
        down = samplerate // gcd
        
        # Resample using polyphase filtering (high quality)
        resampled_data = resample_poly(data, up, down, axis=0)
        
        # Save new file
        sf.write(output_path, resampled_data, target_rate)
        print(f"Upsampled from {samplerate} Hz to {target_rate} Hz → saved to {output_path}")
    
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python upsample_wav.py <input.wav> <output.wav> <target_rate>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    try:
        target_rate = int(sys.argv[3])
        if target_rate <= 0:
            raise ValueError("Target rate must be positive.")
    except ValueError:
        print("Error: target_rate must be a positive integer.")
        sys.exit(1)
    
    upsample_wav(input_file, output_file, target_rate)
