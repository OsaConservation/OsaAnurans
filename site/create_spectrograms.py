import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

species_path = "../specieslist.csv"

df = pd.read_csv(species_path, sep=";")
df = df.loc[df["display"] == 1]

for species in list(df["scientific_name"]):

    try:
        y,sr = librosa.load("./audio/" + species + ".mp3")
        D = librosa.stft(y)

        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)


        fig, ax = plt.subplots(figsize=(10,4))
        fig.patch.set_visible(False)
        ax.axis("off")
        librosa.display.specshow(S_db, sr=sr)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.savefig("./img/spec/" + species, transparent=True, bbox_inches="tight", pad_inches=0)
    except FileNotFoundError:
        print("Error:" + species + "does not have an audio file.")
