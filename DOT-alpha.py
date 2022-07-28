'''
DOT | alpha version

Done:) 1. Recognize my voice and other voices
Done:) 2. Speech Recognition with own dataset and own all "Transformers NN" 
3. chatbot Transformers NN implementation "add response to conversational AI", by my own
4. voice to the reponses 
5. Calculate the perfect parameters for all the models with math of CNN and TNN
6. DOT can explain difficult things "easy" phase two of dot
'''
from tensorflow.keras.models import load_model
import speech_recognition as sr
from scipy import signal
import numpy as np
import pyaudio
import threading


voice_model = load_model('D:/DOT/Voiceclassification/voiceclassify2.model')
classes = ['other', 'Bernardo']

CHUNKSIZE = 1024
rate = 240000 

# Initialize the audio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=CHUNKSIZE)

r = sr.Recognizer()

while 1:
	data = stream.read(rate) # prints weird numbers
	numpydata = np.frombuffer(data, dtype=np.int16)
	frecuencies, times, spectrogram = signal.spectrogram(numpydata, rate)
	
	data = np.array([np.expand_dims(spectrogram, -1)])

	audio_pred = numpydata.reshape(1, -1)
			
	vc_prediction = voice_model.predict(data)[0]
	idx = np.argmax(vc_prediction)
	label = classes[idx]

	print(label, " | ", vc_prediction[idx]*100, "%")
	
	if label == "Bernardo":
		with sr.Microphone() as source:
			print("listening...")
			audio = r.listen(source)
		try: 
			print(r.recognize_google(audio))
			break
		except:
			print("Could not understand you, try again")
			continue
