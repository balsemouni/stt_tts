# un AGC (Automatic Gain Control), soit un Contrôle Automatique du Gain.
# Prevents sound from being too loud or too quiet.
import numpy as np

class SimpleAGC:
    """
    Lightweight AGC for speech
    """
    def __init__(self, target_rms=0.02, max_gain=10.0):
        #RMS (Root Mean Square) qui est une mesure de la puissance moyenne du son.
        self.target_rms = target_rms
        #C'est une sécurité. On ne veut pas multiplier le son par plus de 10,
        # car sinon on amplifierait aussi énormément le bruit de fond (souffle) quand il n'y a pas de parole.
        self.max_gain = max_gain
    #Calcul de l'énergie actuelle
    def process(self, audio):
        rms = np.sqrt(np.mean(audio**2) + 1e-8)
        if rms < 1e-5:
            return audio

        gain = min(self.target_rms / rms, self.max_gain)
        return audio * gain
