import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import soundfile as sf
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from scipy.signal import butter, sosfilt
from sklearn.preprocessing import MinMaxScaler

#definition des constantes et fichiers
dim = 128  # Dimension du spectrogramme
musique_a_traiter = "A Classic Education - NightOwl.stem.wav"
fichier_test_vocals = "A Classic Education - NightOwl.stem_vocals.wav"


def creer_spectrogramme(fichier_wav, afficher=False):
    """
    Crée et retourne le spectrogramme d'un fichier audio WAV.
    Args:
        fichier_wav (str): Chemin vers le fichier audio WAV.
        afficher (bool): Si True, affiche le spectrogramme. Par défaut, False.
    Returns:
        tuple: Un tuple contenant :
            - S_db_resized (ndarray): Le spectrogramme normalisé et redimensionné en décibels.
            - S_phase (ndarray): La phase du spectrogramme.
    """
    # Charger le fichier audio
    y, sr = librosa.load(fichier_wav, sr=None)
    
    # Calculer le spectrogramme
    S = librosa.stft(y)
    S_amplitude, S_phase = librosa.magphase(S)
    
    # Convertir le spectrogramme en décibels
    S_db = librosa.amplitude_to_db(S_amplitude, ref=np.max)

    # Normaliser le spectrogramme avec MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    S_db_normalized = scaler.fit_transform(S_db)  # Applique la normalisation
    
    # Redimensionner le spectrogramme à la taille (dim, dim)
    zoom_factors = (dim / S_db_normalized.shape[0], dim / S_db_normalized.shape[1])
    S_db_resized = zoom(S_db_normalized, zoom_factors, order=3)
    
    if afficher:
        # Afficher le spectrogramme
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db_resized, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogramme (dB)')
        plt.tight_layout()
        plt.show()
    
    return S_db_resized, S_phase

def get_shape_de_base(fichier_wav):
    """
    Charge un fichier audio WAV, calcule le spectrogramme STFT et retourne l'amplitude du spectrogramme ainsi que sa forme.

    Paramètres:
    fichier_wav (str): Chemin vers le fichier audio WAV.

    Retours:
    tuple: Un tuple contenant l'amplitude du spectrogramme (numpy.ndarray) et la forme de cette amplitude (tuple).
    """
    y, sr = librosa.load(fichier_wav, sr=None)
    S = librosa.stft(y)
    S_amplitude, S_phase = librosa.magphase(S)
    return S_amplitude, S_amplitude.shape

# Créer un spectrogramme à partir d'un fichier audio

spectrogramme, S_phase = creer_spectrogramme(musique_a_traiter, afficher=False)
spectrogramme_initial_normalise, phase = creer_spectrogramme(musique_a_traiter)
spectrogramme_initial, dim_spec_base = get_shape_de_base(musique_a_traiter)
original_shape = dim_spec_base
print(f"Shape du spectrogramme d'entree : {original_shape}")
print(f"Shape du spectrogramme redim. et normalisé: {spectrogramme.shape}")

# modele Unet
def build_unet(input_shape):
    """
    Construit un modèle U-Net pour la segmentation d'images.
    Args:
        input_shape (tuple): La forme des entrées du modèle, par exemple (hauteur, largeur, canaux).
    Returns:
        keras.Model: Le modèle U-Net construit.
    Le modèle U-Net est une architecture de réseau de neurones convolutifs utilisée principalement pour la segmentation d'images. 
    Il est composé de couches de convolution, de pooling et d'upsampling, avec des connexions de concaténation entre les couches correspondantes de l'encodeur et du décodeur.
    Exemple:
        model = build_unet((128, 128, 1))
        model.summary()
    """
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

    up2 = UpSampling2D((2, 2))(conv3)
    up2 = Concatenate()([up2, conv2])
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    
    up1 = UpSampling2D((2, 2))(conv4)
    up1 = Concatenate()([up1, conv1])
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv5)
    #outputs = Conv2D(1, (1, 1), activation='linear', padding='same')(conv5)

    
    model = Model(inputs, outputs)
    print("U-Net Model built")
    return model

def creer_x_trains_vocals():
    """
    Crée un tableau de spectrogrammes à partir des fichiers audio dans un répertoire spécifique.

    Cette fonction parcourt tous les fichiers dans le répertoire "/media/cytech/ONE-DISK/trainF/segments_vocals",
    vérifie si chaque fichier se termine par ".wav", puis crée un spectrogramme pour chaque fichier audio trouvé.
    Les spectrogrammes sont ensuite ajoutés à une liste qui est convertie en un tableau NumPy.

    Returns:
        np.ndarray: Un tableau NumPy contenant les spectrogrammes des fichiers audio.
    """
    X_trains = []
    # for file in sorted(os.listdir("train/vocals")):
    for file in sorted(os.listdir("/media/cytech/ONE-DISK/trainF/segments_vocals")):
        if file.endswith(".wav"):
            spectrogramme, phase = creer_spectrogramme(os.path.join("/media/cytech/ONE-DISK/trainF/segments_vocals", file))
            X_trains.append(spectrogramme)
    return np.array(X_trains)

def creer_x_trains_instrumental():
    """
    Crée un tableau de spectrogrammes à partir des fichiers audio dans un répertoire spécifique.

    Cette fonction parcourt tous les fichiers dans le répertoire "/media/cytech/ONE-DISK/trainF/segments_instru",
    vérifie si chaque fichier se termine par ".wav", puis crée un spectrogramme pour chaque fichier audio valide.
    Les spectrogrammes sont ensuite ajoutés à une liste qui est convertie en un tableau NumPy.

    Returns:
        np.ndarray: Un tableau contenant les spectrogrammes des fichiers audio.
    """
    X_trains = []
    # for file in sorted(os.listdir("train/instrumental")):
    for file in sorted(os.listdir("/media/cytech/ONE-DISK/trainF/segments_instru")):
        if file.endswith(".wav"):
            spectrogramme, phase = creer_spectrogramme(os.path.join("/media/cytech/ONE-DISK/trainF/segments_instrumental", file))
            X_trains.append(spectrogramme)
    return np.array(X_trains)

def creer_y_trains():
    """
    Crée une liste de spectrogrammes à partir des fichiers audio dans un répertoire spécifique.

    Cette fonction parcourt tous les fichiers .wav dans le répertoire "/media/cytech/ONE-DISK/trainF/segments_mix",
    crée un spectrogramme pour chaque fichier et ajoute ce spectrogramme à une liste. La liste des spectrogrammes
    est ensuite convertie en un tableau numpy et retournée.

    Returns:
        np.ndarray: Un tableau numpy contenant les spectrogrammes des fichiers audio.
    """
    Y_trains = []
    # for file in sorted(os.listdir("train/mix")):
    for file in sorted(os.listdir("/media/cytech/ONE-DISK/trainF/segments_mix")):
        if file.endswith(".wav"):
            spectrogramme, phase = creer_spectrogramme(os.path.join("/media/cytech/ONE-DISK/trainF/segments_mix", file))
            Y_trains.append(spectrogramme)
    return np.array(Y_trains)


# Entraîner le modèle (si fichier .keras non trouvé)

# x_trains = creer_x_trains_vocals()
# y_trains = creer_y_trains()
# # Normalisation des spectrogrammes
# x_trains = (x_trains - x_trains.min()) / (x_trains.max() - x_trains.min())
# y_trains = (y_trains - y_trains.min()) / (y_trains.max() - y_trains.min())

# # Ajouter une dimension de profondeur (1 canal) pour les entrées du modèle
# x_trains = np.expand_dims(x_trains, axis=-1)
# y_trains = np.expand_dims(y_trains, axis=-1)


# input_shape = (dim, dim, 1)
# model = build_unet(input_shape)
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
# model.fit(x_trains, y_trains, epochs=10, batch_size=8)
# model.save("unet_model_vocals_seg.keras")

# Charger le modèle
model = tf.keras.models.load_model("unet_model_vocals.keras")

# Prédire le spectrogramme
spectrogramme_predit = model.predict(np.expand_dims(spectrogramme, axis=0))[0]


# Afficher le spectrogramme prédit
plt.figure(figsize=(10, 4))
spectrogramme_predit = np.squeeze(spectrogramme_predit, axis=-1)
print(f"Shape du spectrogramme prédit: {spectrogramme_predit.shape}")
librosa.display.specshow(spectrogramme_predit, sr=44100, x_axis='time', y_axis='log', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme Prédit (dB)')
plt.tight_layout()
plt.show()

S_test = creer_spectrogramme(fichier_test_vocals, afficher=True)

# Comparer le spectrogramme prédit avec le spectrogramme original
print("Comparaison des spectrogrammes")
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
librosa.display.specshow(spectrogramme, sr=44100, x_axis='time', y_axis='log', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme Original (dB)')
plt.subplot(1, 3, 2)

librosa.display.specshow(spectrogramme_predit, sr=44100, x_axis='time', y_axis='log', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme Prédit (dB)')
plt.subplot(1, 3, 3)
librosa.display.specshow(S_test[0], sr=44100, x_axis='time', y_axis='log', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme Test (dB)')
plt.tight_layout()
plt.show()


# def restaurer_spectrogramme(spectrogramme_normalized, original_shape):
#     # Dé-normaliser le spectrogramme
#     S_db_resized = spectrogramme_normalized * (np.max(spectrogramme_normalized) - np.min(spectrogramme_normalized)) + np.min(spectrogramme_normalized)
    
#     # Redimensionner le spectrogramme à sa taille originale
#     S_db = zoom(S_db_resized, (original_shape[0] / 128, original_shape[1] / 128))
    
#     return S_db

# def restaurer_spectrogramme(spectrogramme_normalized, original_shape, original_min, original_max):
#     # Revenir à l'échelle originale
#     S_db_resized = spectrogramme_normalized * (original_max - original_min) + original_min
    
#     # Redimensionner à la forme originale
#     S_db = zoom(S_db_resized, (original_shape[0] / dim, original_shape[1] / dim))
#     return S_db


# # Restaurer le spectrogramme prédit à sa forme originale et à son échelle
# spectrogramme_separe = restaurer_spectrogramme(
#     spectrogramme_predit,
#     original_shape=original_shape,
#     original_min=np.min(spectrogramme_initial),
#     original_max=np.max(spectrogramme_initial)
# )


# # Afficher le spectrogramme restauré
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(spectrogramme_separe, sr=44100, x_axis='time', y_axis='log', cmap='viridis')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogramme Restauré (dB)')
# plt.tight_layout()
# plt.show()


# def reconstruire_audio(spectrogramme, phase, sr=44100):
#     # Convertir le spectrogramme normalisé en amplitude réelle
#     S_amplitude = librosa.db_to_amplitude(spectrogramme)
    
#     # Recréer le spectrogramme complexe
#     S_complex = S_amplitude * phase
    
#     # Reconstruire le signal audio
#     y = librosa.istft(S_complex)
#     return y


# # Recréer le fichier audio
# y = reconstruire_audio(spectrogramme_separe, S_phase)

# def normalisation_audio(audio):
#     max_amplitude = np.max(np.abs(audio))
#     return audio / max_amplitude

# # audio_normalise = normalisation_audio(y)


# def filtre_passe_bande(audio, lowcut=85.0, highcut=3000.0, sr=44100, order=4):
#     sos = butter(order, [lowcut, highcut], btype='band', fs=sr, output='sos')
#     return sosfilt(sos, audio)

# audio_filtre = filtre_passe_bande(y, lowcut=300, highcut=3000.0)


# # Sauvegarder le fichier audio
# sf.write(f'{musique_a_traiter}_separe.wav', y, 44100)

