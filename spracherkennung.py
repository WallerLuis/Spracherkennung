"""
Created on Tue May 25 15:06:16 2021

@author: Luis

Important stats:

    epoch: Numbers of time the neuronal network is training
    path: path to dataset
    
    Functions:
        __init__(): constructor: by starting the programm starting data_arrangment() and plot_graphs()
        data_arrangment(): arranging data in test and training data.
        create_model(): compiling a deep neural network
        plot_graphs(): plotting graph1(Training acurazy) and confusion graph for test data 
        test(): testing one data file. 
        plot_mel_MFCC(audio_file): evaluating a MFCC on a wavfile. But only wav files above 600kb
    """
    
import numpy as np
import scipy.signal
import zaf
import matplotlib.pyplot as plt
import os
import librosa
import random as rn
import warnings
with warnings.catch_warnings(): # for warnings spam by tensorflow
    warnings.filterwarnings("ignore",category=FutureWarning) 
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
import seaborn as sns
import speech_recognition as sr
import shutil

class Spracherkennung():
        
    def __init__(self): 
        self.data_arrangment()
        self.plot_graphs()
    
    epoch = 100
    path = 'Audio/recordings'
        
    
    def data_arrangment(self):
             
        print("Arranging data:")

        #Shuffling the files for better performance
        filenames=[]
        for flist in os.listdir(self.path):
            filenames.append(flist)
        rn.shuffle(filenames)
        print(f"Number of samples in databank: {len(filenames)}")  
        
        def split_to_percentage(data,percentage):
            return  data[0: int(len(data)*percentage)] , data[int(len(data)*percentage):]
        
        train_data,test_data = split_to_percentage(filenames,0.8)
        
        #creating the train and test data from the MNIST audio dataset.
        train_mfccs = []
        train_y = []
        test_mfccs = []
        self.test_y = []
        self.pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a,
                            np.zeros((a.shape[0],i - a.shape[1]))))
  
        for i in range(len(train_data)): #for train data
                
            struct = train_data[i].split('_')
            digit = struct[0]
            wav, sr = librosa.load(os.path.join(self.path , train_data[i]))
            mfcc = librosa.feature.mfcc(wav)
            padded_mfcc = self.pad2d(mfcc,40)
            train_mfccs.append(padded_mfcc)
            train_y.append(digit)

        for i in range(len(test_data)): #for test data
    
            struct = test_data[i].split('_')
            digit = struct[0]
            wav, sr = librosa.load(os.path.join(self.path , test_data[i]))
            mfcc = librosa.feature.mfcc(wav)
            padded_mfcc = self.pad2d(mfcc,40)
            test_mfccs.append(padded_mfcc)
            self.test_y.append(digit)


        train_mfccs = np.array(train_mfccs)
        train_y = to_categorical(np.array(train_y))
        test_mfccs = np.array(test_mfccs)
        self.test_y = to_categorical(np.array(self.test_y))
        self.train_X_ex = np.expand_dims(train_mfccs, -1)
        self.test_X_ex = np.expand_dims(test_mfccs, -1)

        def create_model():
            
            #Create a deep neural network
            print(f"creating a deep neutral network with {self.epoch} epochs")
            ip = tf.keras.Input(shape=self.train_X_ex[0].shape)
            m = tf.keras.layers.Conv2D(filters=256, kernel_size=(4, 4), activation='relu')(ip)
            m = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(m)
            m = tf.keras.layers.BatchNormalization()(m)
            m = tf.keras.layers.Dropout(0.2)(m)
            m = tf.keras.layers.Flatten()(m)
            m = tf.keras.layers.Dense(64, activation='relu')(m)
            m = tf.keras.layers.Dense(32, activation='relu')(m)
            op = tf.keras.layers.Dense(10, activation='softmax')(m)
            self.model = tf.keras.Model(inputs=ip, outputs=op)
            
            self.model.summary()
                
            #Compile and fit the neural network
            self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
            return self.model
        
        #creating model
        create_model()

        # Create a callback that saves the model's weights
        checkpoint_path = "cp.ckpt"
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        save_best_only=True, mode='max', monitor='val_accuracy', verbose=1)
        
        self.history = self.model.fit(self.train_X_ex,
              train_y,
              epochs = self.epoch,
              batch_size=32,
              validation_data=(self.test_X_ex, self.test_y),
              callbacks=[cp_callback])
        
        print("Saving the deep neural network:")
        
        loss, acc = self.model.evaluate(self.test_X_ex, self.test_y, verbose=2)
        print("Restored last model of training, accuracy: {:5.2f}%".format(100 * acc))
          
        # Loads the weights
        self.model.load_weights(checkpoint_path).expect_partial()   
        
        # Re-evaluate the model
        loss, acc = self.model.evaluate(self.test_X_ex, self.test_y, verbose=2)
        print("Restored ceckpoint model, accuracy: {:5.2f}%".format(100 * acc))
        
        print("finish with initialing. Ready for tests :)")
        
    def test(self, test_file):#mit eigenen Dateien testen
    
        print("Test:")
        print(f"Testfile: {test_file}")
        
        own_mfccs = []
        own_wav, sr = librosa.load(os.path.join(self.path , test_file))
        own_mfcc = librosa.feature.mfcc(own_wav)
        own_padded_mfcc = self.pad2d(own_mfcc,40)
        own_mfccs.append(own_padded_mfcc) 
        own_X_ex = np.expand_dims(own_mfccs, -1)
        
        predictions = self.model.predict(own_X_ex)
        predictions = predictions.astype(int)
        
        for i in predictions:
            if i.any() != 0 :
                print(f"Predicted number: {np.argmax(predictions)}")
            else:
                print("System couldn't predict a number")
        
    def plot_graphs(self):
        
        print("Plotting graphs:")
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        #Display the Confusion Matrix for test data
        y_pred= self.model.predict(self.test_X_ex)
        y_p= np.argmax(y_pred, axis=1)
        y_pred=y_pred.astype(int)
        y_t=np.argmax(self.test_y, axis=1)
        confusion_mtx = tf.math.confusion_matrix(y_t, y_p) 
        plt.figure(figsize=(5, 5))
        sns.heatmap(confusion_mtx, 
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()

    speech_engine = sr.Recognizer()
    def from_microphone(self):
        with sr.Microphone() as micro:
            print("Recording...")
            self.audio = self.speech_engine.record(micro, duration=2)
           
            # write audio to a WAV file
            with open("Audio/recordings/output.wav", "wb") as f:
                self.audio = f.write(self.audio.get_wav_data())
           
                print("Recognition...")
                SE.test("output.wav")
  
        os.remove("Audio/recordings/output.wav")
        
    def plot_mel_MFCC(audio_file):
        # Read the audio signal (normalized) with its sampling frequency in Hz, and average it over its channels
        audio_signal, sampling_frequency = zaf.wavread(audio_file)
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
        
SE = Spracherkennung()
SE.test("5_jackson_7.wav")