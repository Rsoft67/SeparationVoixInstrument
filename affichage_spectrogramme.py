import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import soundfile as sf

def display_spectrogram(filepath):
    """
    Affiche le spectrogramme d'un fichier audio donné.
    Paramètres:
    -----------
    filepath : str
        Le chemin vers le fichier audio à analyser.
    Cette fonction charge le fichier audio spécifié, calcule son spectrogramme
    et l'affiche en utilisant une échelle logarithmique pour l'axe des fréquences.
    Exemples:
    ---------
    >>> display_spectrogram('/chemin/vers/fichier_audio.wav')
    """
    
    # Chargement du fichier audio
    y, sr = librosa.load(filepath, sr=None)
    
    # Calcul du spectrogramme
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    # Affichage du spectrogramme
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogramme')
    plt.show()

# Afficher le spectrogramme du fichier audio
input_path = "A Classic Education - NightOwl.stem_segment_3.wav"
plt.figure(figsize=(15, 10))

# Afficher le spectrogramme du premier fichier audio
plt.subplot(3, 1, 1)
y, sr = librosa.load(input_path, sr=None)
S = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme - ' + input_path)

# Afficher le spectrogramme du deuxième fichier audio
plt.subplot(3, 1, 2)
y, sr = librosa.load("instruSpleeterClara.wav", sr=None)
S = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme - Instrumental par Spleeter')

# Afficher le spectrogramme du troisième fichier audio
plt.subplot(3, 1, 3)
y, sr = librosa.load("vocalsSpleeterClara.wav", sr=None)
S = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme - Voix par Spleeter')

plt.tight_layout()
plt.show()
