import librosa
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write
import IPython.display as ipd

# Charger l'audio
def chargerAudio(file_path):
    """
    Charge un fichier audio et retourne le signal audio et le taux d'échantillonnage.

    Args:
        file_path (str): Le chemin du fichier audio à charger.

    Returns:
        tuple: Un tuple contenant le signal audio (numpy.ndarray) et le taux d'échantillonnage (int).
    """
    y, sr = librosa.load(file_path, sr=None)  # y = signal audio, sr = taux d'échantillonnage
    return y, sr

# Traitement de l'audio
def traitement2(song_path):
    """
    Traite un fichier audio pour séparer les parties vocales et instrumentales.

    Args:
        song_path (str): Le chemin vers le fichier audio à traiter.

    Returns:
        tuple: Un tuple contenant deux éléments :
            - y_foreground (numpy.ndarray): Le signal audio des parties vocales.
            - y_background (numpy.ndarray): Le signal audio des parties instrumentales.

    Cette fonction effectue les étapes suivantes :
    1. Charge le fichier audio spécifié.
    2. Calcule la durée totale de l'audio en échantillons et en secondes.
    3. Définit un intervalle de 20 secondes maximum à partir de 90 secondes si possible, sinon depuis le début.
    4. Extrait et affiche l'intervalle audio défini.
    5. Calcule le spectrogramme de l'audio.
    6. Applique un filtrage pour séparer les composantes vocales et instrumentales.
    7. Crée des masques pour les parties vocales et instrumentales.
    8. Extrait les parties vocales et instrumentales du spectrogramme.
    9. Reconstruit les signaux audio pour les parties vocales et instrumentales.
    10. Ajuste les indices pour les fichiers courts.
    11. Affiche l'intervalle audio filtré.
    12. Sauvegarde les fichiers audio des parties vocales et instrumentales sous les noms "foreground.wav" et "background_only.wav".
    """
    # Charger l'audio
    y, sr = chargerAudio(song_path)

    # Calculer la durée en échantillons
    audio_length = len(y)
    print(f"Durée totale : {audio_length / sr:.2f} secondes ({audio_length} échantillons)")

    # Définir un intervalle de 20 secondes maximum, ou moins si l'audio est plus court
    start = 90 * sr if audio_length > 90 * sr else 0  # Départ à 90 secondes si possible, sinon début
    end = start + (20 * sr) if audio_length >= start + 20 * sr else audio_length  # 20s ou jusqu'à la fin

    # Extraire l'échantillon à lire
    print(f"Lecture de l'intervalle : de {start / sr:.2f} à {end / sr:.2f} secondes")
    ipd.display(ipd.Audio(data=y[start:end], rate=sr))

    # Spectrogramme
    transformee_fourier = librosa.stft(y)
    S_full, phase = librosa.magphase(transformee_fourier)

    # Filtrage
    S_filter = librosa.decompose.nn_filter(
        S_full, aggregate=np.median, metric="cosine", width=int(librosa.time_to_frames(2, sr=sr))
    )
    S_filter = np.minimum(S_full, S_filter)

    # Séparation voix/instrumental
    margin_i, margin_v = 5, 11
    power = 3

    # Créer les masques
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
    mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)

    # Extraire les parties
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    # Reconstruction des signaux audio
    y_foreground = librosa.istft(S_foreground * phase)
    y_background = librosa.istft(S_background * phase)

    # Ajuster les indices pour les fichiers courts
    start = min(start, len(y_foreground))
    end = min(end, len(y_foreground))
    print(f"Lecture audio filtré : de {start / sr:.2f} à {end / sr:.2f} secondes")
    ipd.display(ipd.Audio(data=y_foreground[start:end], rate=sr))

    # Sauvegarder les fichiers audio
    write("foreground.wav", sr, (y_foreground * 32767).astype(np.int16))
    write("background_only.wav", sr, (y_background * 32767).astype(np.int16))

    return y_foreground, y_background

# Chemin du fichier audio
# input_path = "A Classic Education - NightOwl.stem_segment_3.wav"
input_path = "Actions - South Of The Water.stem_segment_6.wav"
input_path = "Clara Berry And Wooldog - Air Traffic.stem_segment_9.wav"
#input_path = "musique.wav"
y_foreground, y_background = traitement2(input_path)
