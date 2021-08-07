# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:06:16 2021

@author: Luis
"""
import numpy as np
import scipy.signal
import zaf
import matplotlib.pyplot as plt
import os
import glob
from scipy.io import wavfile 
import wave

# testdateien
file = str("audio_file.wav")
file2 = str("0_test_0.wav")
file3 = str("0_jackson_0.wav")

# Databank in List
databank = []
path = 'Audio/recordings'
                    
for filename in glob.glob(os.path.join(path, '*.wav')):
    w = wave.open(filename, 'rb')
    d = w.readframes(w.getnframes())
    databank.append(d)
    w.close()

#andere Möglichkeit die Liste zu füllen
#files = os.listdir(path)

#for filename in glob.glob(os.path.join(path, '*.wav')):
 #   samplerate, data = wavfile.read(filename)
  #  databank.append(data)
    
#in array statt list 
np_databank = np.array(databank)

#länge der Datenbank
print(len(np_databank))

def plot_mel_MFCC(audio_signal):
    # Read the audio signal (normalized) with its sampling frequency in Hz, and average it over its channels
    audio_signal, sampling_frequency = zaf.wavread(file)# file nur zum Testen (ich wechsel immer zwischen 1-3), eigentlich audio_signal aber das gibt einen Fehler
    audio_signal = np.mean(audio_signal, 1)

    # Set the parameters for the Fourier analysis
    window_length = pow(2, int(np.ceil(np.log2(0.04*sampling_frequency))))
    window_function = scipy.signal.hamming(window_length, sym=False)
    step_length = int(window_length/2)
    
    # Compute the mel filterbank
    number_mels = 40
    mel_filterbank = zaf.melfilterbank(sampling_frequency, window_length, number_mels)

    # Compute the MFCCs using the filterbank
    number_coefficients = 20
    audio_mfcc = zaf.mfcc(audio_signal, window_function, step_length, mel_filterbank, number_coefficients)

    # Compute the mel spectrogram using the filterbank
    mel_spectrogram = zaf.melspectrogram(audio_signal, window_function, step_length, mel_filterbank)
    
    # Display the MFCCs in seconds and the mel spectrogram in dB, seconds, and Hz
    number_samples = len(audio_signal)
    xtick_step = 1
    plt.figure(figsize=(17, 10))
    
    #mel spectogram
    zaf.melspecshow(mel_spectrogram, number_samples, sampling_frequency, window_length, xtick_step)
    plt.title("Mel spectrogram (dB)")
    plt.show()
    
    #MFCC
    plt.subplot(3, 1, 1), zaf.mfccshow(audio_mfcc, number_samples, sampling_frequency, xtick_step), plt.title("MFCCs")
    plt.show()

#ganze Liste durchführen
#for i in databank:
#    plot_mel_MFCC(np_databank[i])
plot_mel_MFCC(np_databank[1])