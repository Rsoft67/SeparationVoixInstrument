import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from pydub import AudioSegment
from pydub.playback import play
import threading
import librosa
import numpy as np
from scipy.io.wavfile import write
import os

class Separateur:
    def __init__(self, root):
        self.root = root
        self.root.title("Traitement signal : Séparateur de musique")
        self.root.geometry("1000x550")
        self.root.config(bg="#F0F0F0") 

        # Bloc qui gère les infos du haut de l'écran
        self.header_frame = tk.Frame(root, bg="#F0F0F0")
        self.header_frame.pack(fill="x", pady=10)

        # Prénoms des membres du groupe
        self.names_label = tk.Label(self.header_frame, text="Membres du groupe:\nLorion Benjamin\nPointeau Martin\nSeddiki Rayane\nWohl Nathan\nKoniarz Tobias\nKhatib Robin\nMedrano Oscar", 
                                    font=("Arial", 12), bg="#F0F0F0", fg="#2C3E50")
        self.names_label.pack(side="left", padx=10)

        #Prof classe année
        self.class_info_label = tk.Label(self.header_frame, text="M. Bahtiti | ING2 MI2 | 2024-2025", 
                                         font=("Arial", 12), bg="#F0F0F0", fg="#2C3E50")
        self.class_info_label.pack(side="right", padx=10)

        # Titre central
        title = tk.Label(root, text="Séparation de la Musique", font=("Arial", 20, "bold"), bg="#F0F0F0", fg="#2C3E50")
        title.pack(pady=15)

        # Bouton pour charger une musique
        self.load_button = tk.Button(root, text="Charger une musique", command=self.load_music, 
                                     font=("Arial", 12), bg="#2980B9", fg="white", relief="flat", width=20)
        self.load_button.pack(pady=10)

        # Barre de progression
        self.progress = Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)

        # Boutons de lecture
        button_style = {"font": ("Arial", 12), "bg": "#27AE60", "fg": "white", "relief": "flat", "width": 20}
        
        self.play_instrumental_button = tk.Button(root, text="Jouer l'instrumentale", command=self.play_instrumental, state=tk.DISABLED, **button_style)
        self.play_instrumental_button.pack(pady=5)

        self.play_vocals_button = tk.Button(root, text="Jouer les paroles", command=self.play_vocals, state=tk.DISABLED, **button_style)
        self.play_vocals_button.pack(pady=5)

        self.play_original_button = tk.Button(root, text="Jouer la musique originale", command=self.play_original, state=tk.DISABLED, **button_style)
        self.play_original_button.pack(pady=5)

        # Variables pour stocker les chemins des fichiers audio
        self.instrumental_path = None #l'instru uniquement
        self.vocals_path = None #les paroles uniquement
        self.original_audio = None  # Pour stocker l'audio original
        self.stop_playback = False  # permet de gérer la "file d'attente" par défaut sur false car aucun audio est en cour
        self.playback_thread = None  # variable qui stock si y'a une lecture en cours ou non


    def load_music(self):
        # Ouvrir le sélecteur de fichiers pour choisir un fichier audio
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
        if file_path:
            try:
                self.progress["value"] = 10
                self.root.update_idletasks()

                # Charger et séparer l'audio
                y, sr = self.charger_audio(file_path)
                if y is not None:
                    self.progress["value"] = 50
                    self.root.update_idletasks()

                    self.vocals_path, self.instrumental_path = self.separation_audio(y, sr)# appelle la fonction pour obtenir l'audio et l'instru séparé
                    self.original_audio = AudioSegment.from_file(file_path)

                    self.progress["value"] = 100
                    self.root.update_idletasks()

                    # Active les boutons de lecture car ils sont désactivés de base
                    self.play_instrumental_button.config(state=tk.NORMAL)
                    self.play_vocals_button.config(state=tk.NORMAL)
                    self.play_original_button.config(state=tk.NORMAL)

            except Exception as e:
                messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")
                self.progress["value"] = 0

    def charger_audio(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=None)  # y = signal audio, sr = taux d'échantillonnage
            return y, sr
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le fichier : {e}")
            return None, None

    def separation_audio(self, y, sr):
        # Spectrogramme
        transformee_fourier = librosa.stft(y)
        S_full, phase = librosa.magphase(transformee_fourier)

        # Filtrage
        S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sr)))
        S_filter = np.minimum(S_full, S_filter)

        # Création des masques
        margin_i, margin_v = 5, 11
        power = 3

        mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
        mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)

        # Extraire la voix et l'instrumental
        S_foreground = mask_v * S_full
        S_background = mask_i * S_full

        # Reconstruction du signal audio
        y_foreground = librosa.istft(S_foreground * phase)
        y_background = librosa.istft(S_background * phase)

        # Sauvegarde des fichiers audio séparés
        vocals_path = os.path.join('vocals_only.wav')
        instrumental_path = os.path.join('instrumental_only.wav')

        write(vocals_path, sr, (y_foreground * 32767).astype(np.int16))
        write(instrumental_path, sr, (y_background * 32767).astype(np.int16))

        return vocals_path, instrumental_path

    def play_original(self):
        if self.original_audio:
            self.stop_audio()  # pour savoir si on peut le jouer ou si il est en file d'attente
            self.playback_thread = threading.Thread(target=self._play_audio, args=(self.original_audio,))
            self.playback_thread.start()

    def play_instrumental(self):
        if self.instrumental_path:
            instrumental = AudioSegment.from_wav(self.instrumental_path)
            self.stop_audio()  # pour savoir si on peut le jouer ou si il est en file d'attente
            self.playback_thread = threading.Thread(target=self._play_audio, args=(instrumental,))
            self.playback_thread.start()

    def play_vocals(self):
        if self.vocals_path:
            vocals = AudioSegment.from_wav(self.vocals_path)
            self.stop_audio()  # pour savoir si on peut le jouer ou si il est en file d'attente
            self.playback_thread = threading.Thread(target=self._play_audio, args=(vocals,))
            self.playback_thread.start()

    def _play_audio(self, audio):
        self.stop_playback = False
        play(audio)

    def stop_audio(self):
        if self.playback_thread and self.playback_thread.is_alive():
            self.stop_playback = True
            self.playback_thread.join()  # Attendre que l'audio en cours soit arrêté
            self.playback_thread = None  # Réinitialiser le thread

# Lancer l'application
if __name__ == "__main__":
    root = tk.Tk()
    app = Separateur(root)
    root.mainloop()
