# import os
# import time
# import pygame
# from gtts import gTTS
# import streamlit as st
# import speech_recognition as sr
# from googletrans import LANGUAGES, Translator

# # Initialize global variables
# isTranslateOn = False
# translator = Translator()  # Initialize the translator module.
# pygame.mixer.init()  # Initialize the mixer module.

# # Create a mapping between language names and language codes
# language_mapping = {name: code for code, name in LANGUAGES.items()}

# def get_language_code(language_name):
#     return language_mapping.get(language_name, language_name)

# def translator_function(spoken_text, from_language, to_language):
#     return translator.translate(spoken_text, src='{}'.format(from_language), dest='{}'.format(to_language))

# def text_to_voice(text_data, to_language):
#     myobj = gTTS(text=text_data, lang='{}'.format(to_language), slow=False)
#     myobj.save("cache_file.mp3")
#     audio = pygame.mixer.Sound("cache_file.mp3")  # Load a sound.
#     audio.play()
#     os.remove("cache_file.mp3")

# def main_process(output_placeholder, from_language, to_language):
    
#     global isTranslateOn
    
#     while isTranslateOn:
#         rec = sr.Recognizer()
#         with sr.Microphone() as source:
#             output_placeholder.markdown("<h3 style='color:blue;'>🎙️ Listening...</h3>", unsafe_allow_html=True)
#             rec.pause_threshold = 1
#             audio = rec.listen(source, phrase_time_limit=10)
        
#         try:
#             output_placeholder.markdown("<h3 style='color:orange;'>⏳ Processing...</h3>", unsafe_allow_html=True)
#             spoken_text = rec.recognize_google(audio, language='{}'.format(from_language))
            
#             output_placeholder.markdown("<h3 style='color:green;'>🌍 Translating...</h3>", unsafe_allow_html=True)
#             translated_text = translator_function(spoken_text, from_language, to_language)

#             output_placeholder.markdown(f"<h3 style='color:green;'>🔊 Translation: {translated_text.text}</h3>", unsafe_allow_html=True)
#             text_to_voice(translated_text.text, to_language)
    
#         except Exception as e:
#             output_placeholder.error(f"Error: {str(e)}")

# # UI layout
# st.title("🌐 Language Translator")

# # Sidebar for language selection
# st.sidebar.markdown("## Select Languages")

# from_language_name = st.sidebar.selectbox("🎤 Source Language:", list(LANGUAGES.values()), index=list(LANGUAGES.values()).index("english"))
# to_language_name = st.sidebar.selectbox("🔊 Target Language:", list(LANGUAGES.values()), index=list(LANGUAGES.values()).index("spanish"))

# # Convert language names to language codes
# from_language = get_language_code(from_language_name)
# to_language = get_language_code(to_language_name)

# # Instruction text
# st.markdown("""
# <style>
#     .instructions {
#         background-color: black;
#         padding: 15px;
#         border-radius: 10px;
#         margin-bottom: 20px;
#         font-size: 18px;
#     }
# </style>
# <div class="instructions">
#     1. Select the Source and Target languages.<br>
#     2. Press "Start" to begin translation.<br>
#     3. Press "Stop" to end the translation process.
# </div>
# """, unsafe_allow_html=True)

# # Buttons for Start and Stop
# col1, col2 = st.columns(2)
# with col1:
#     start_button = st.button("🟢 Start", use_container_width=True)
# with col2:
#     stop_button = st.button("🔴 Stop", use_container_width=True)

# # Output section
# output_placeholder = st.empty()

# # Check if "Start" button is clicked
# if start_button:
#     if not isTranslateOn:
#         isTranslateOn = True
#         main_process(output_placeholder, from_language, to_language)

# # Check if "Stop" button is clicked
# if stop_button:
#     isTranslateOn = False
#     output_placeholder.markdown("<h3 style='color:red;'>🚫 Translation Stopped</h3>", unsafe_allow_html=True)


# vosk model

# import vosk
# import sounddevice as sd
# import json

# # Path to the Vosk model
# model_path = r"D:\vigy-exp\Vigyaan-Translator-STS-main\vosk-model-small-en-us-0.15"

# # Initialize the Vosk model
# model = vosk.Model(model_path)

# # Initialize the recognizer
# recognizer = vosk.KaldiRecognizer(model, 16000)

# # Callback function to process audio in real time
# def callback(indata, frames, time, status):
#     # Convert the audio data to bytes before passing it to the recognizer
#     if recognizer.AcceptWaveform(bytes(indata)):
#         result = json.loads(recognizer.Result())
#         print(f"Recognized text: {result['text']}")
#     else:
#         print(recognizer.PartialResult())

# # Start recording from the microphone and process the audio stream
# with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
#     print("Listening... Speak now.")
#     sd.sleep(10000)  # Adjust duration as needed (in milliseconds)

# marian nmt

# from transformers import MarianMTModel, MarianTokenizer

# model_name = "Helsinki-NLP/opus-mt-en-hi"
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# model = MarianMTModel.from_pretrained(model_name)

# english_text = "I am happy to meet you"
# tokenized_text = tokenizer.prepare_seq2seq_batch([english_text], return_tensors="pt")
# translated = model.generate(**tokenized_text)
# hindi_translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# print(hindi_translation[0])


# combined
# import vosk
# import sounddevice as sd
# import json
# from transformers import MarianMTModel, MarianTokenizer
# import numpy as np

# # Initialize the Vosk model (Speech-to-Text)
# vosk_model_path = r"D:\vigy-exp\Vigyaan-Translator-STS-main\vosk-model-small-en-us-0.15"
# model = vosk.Model(vosk_model_path)
# recognizer = vosk.KaldiRecognizer(model, 16000)

# # Initialize Marian NMT model (Translation English to Hindi)
# marian_model_name = "Helsinki-NLP/opus-mt-en-hi"
# tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
# translation_model = MarianMTModel.from_pretrained(marian_model_name)

# # Function to translate recognized text using Marian NMT
# def translate_to_hindi(english_text):
#     tokenized_text = tokenizer([english_text], return_tensors="pt", padding=True)
#     translated = translation_model.generate(**tokenized_text)
#     hindi_translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
#     return hindi_translation[0]

# # Callback function to process real-time audio with Vosk
# def callback(indata, frames, time, status):
#     # Ensure indata is a NumPy array and convert it to bytes
#     indata_bytes = np.array(indata).tobytes()  # Convert indata to bytes
#     if recognizer.AcceptWaveform(indata_bytes):
#         result = json.loads(recognizer.Result())
#         recognized_text = result['text']
#         print(f"Recognized text: {recognized_text}")
        
#         if recognized_text:
#             # Translate the recognized text to Hindi
#             hindi_translation = translate_to_hindi(recognized_text)
#             print(f"Translated text (English to Hindi): {hindi_translation}")

# # Start recording audio from the microphone and process the stream
# with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
#     print("Listening... Speak now.")
#     sd.sleep(10000)  # Adjust duration as needed (in milliseconds)

# espeak ng
# import os

# # Set the text and the language to Hindi
# text = "kaise ho tum"
# language = "hi"  # Hindi language code

# # Use os.system to run espeak-ng with the Hindi language setting
# os.system(f'espeak-ng -v {language} "{text}"')

# nmt for hi to en

# from transformers import MarianMTModel, MarianTokenizer

# # Specify the model for translating from Hindi to English
# model_name = "Helsinki-NLP/opus-mt-hi-en"
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# model = MarianMTModel.from_pretrained(model_name)

# # Hindi text to translate
# hindi_text = "मुझे आपसे मिलकर खुशी हुई"

# # Tokenize the input Hindi text
# tokenized_text = tokenizer.prepare_seq2seq_batch([hindi_text], return_tensors="pt")

# # Generate translation (Hindi to English)
# translated = model.generate(**tokenized_text)

# # Decode the translated text
# english_translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# # Print the English translation
# print(english_translation[0])


# vosk model for hindi
# import vosk
# import sounddevice as sd
# import json

# # Path to the Vosk Hindi model
# model_path = r"D:\vigy-exp\Vigyaan-Translator-STS-main\vosk-model-small-hi-0.22"  # Change this to the correct path

# # Initialize the Vosk model for Hindi
# model = vosk.Model(model_path)

# # Initialize the recognizer with the model and sample rate (16000Hz)
# recognizer = vosk.KaldiRecognizer(model, 16000)

# # Callback function to process audio in real time
# def callback(indata, frames, time, status):
#     # Convert the audio data to bytes before passing it to the recognizer
#     if recognizer.AcceptWaveform(bytes(indata)):
#         result = json.loads(recognizer.Result())
#         print(f"Recognized text: {result['text']}")
#     else:
#         print(recognizer.PartialResult())

# # Start recording from the microphone and process the audio stream
# with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
#     print("Listening... Speak now.")
#     sd.sleep(10000)  # Adjust duration as needed (in milliseconds)

# combined for hi to en
# import os
# import vosk
# import sounddevice as sd
# import json
# from transformers import MarianMTModel, MarianTokenizer

# # Path to the Vosk Hindi model
# model_path = r"D:\vigy-exp\Vigyaan-Translator-STS-main\vosk-model-small-hi-0.22"  # Change this to the correct path

# # Initialize the Vosk model for Hindi
# model = vosk.Model(model_path)

# # Initialize the recognizer with the model and sample rate (16000Hz)
# recognizer = vosk.KaldiRecognizer(model, 16000)

# # Specify the model for translating from Hindi to English
# nmt_model_name = "Helsinki-NLP/opus-mt-hi-en"
# tokenizer = MarianTokenizer.from_pretrained(nmt_model_name)
# nmt_model = MarianMTModel.from_pretrained(nmt_model_name)

# # eSpeak-NG function for speaking text in English
# def speak_text(text, language="en"):
#     os.system(f'espeak-ng -v {language} "{text}"')

# # Callback function to process audio in real time
# def callback(indata, frames, time, status):
#     # Convert the audio data to bytes before passing it to the recognizer
#     if recognizer.AcceptWaveform(bytes(indata)):
#         result = json.loads(recognizer.Result())
#         recognized_text = result['text']
#         print(f"Recognized text: {recognized_text}")

#         # Translate recognized Hindi text to English
#         if recognized_text:
#             tokenized_text = tokenizer.prepare_seq2seq_batch([recognized_text], return_tensors="pt")
#             translated = nmt_model.generate(**tokenized_text)
#             english_translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
#             print(f"Translated to English: {english_translation[0]}")

#             # Speak the translated text
#             speak_text(english_translation[0], language="en")

#     else:
#         print(recognizer.PartialResult())

# # Start recording from the microphone and process the audio stream
# with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
#     print("Listening... Speak now.")
#     sd.sleep(10000)  # Adjust duration as needed (in milliseconds)


# # for ui and combine sts from hi to en
# import os
# import json
# import streamlit as st
# import sounddevice as sd
# import vosk
# from transformers import MarianMTModel, MarianTokenizer
# import threading

# # Initialize pygame for audio playback
# import pygame
# pygame.mixer.init()

# # Path to the Vosk Hindi model
# model_path = r"D:\vigy-exp\Vigyaan-Translator-STS-main\vosk-model-small-hi-0.22"  # Change this to the correct path
# model = vosk.Model(model_path)

# # Initialize the recognizer with the model and sample rate (16000Hz)
# recognizer = vosk.KaldiRecognizer(model, 16000)

# # Specify the model for translating from Hindi to English
# nmt_model_name = "Helsinki-NLP/opus-mt-hi-en"
# tokenizer = MarianTokenizer.from_pretrained(nmt_model_name)
# nmt_model = MarianMTModel.from_pretrained(nmt_model_name)

# # Flag to control the translation process
# isTranslateOn = False
# audio_stream = None

# # Function for text-to-speech using eSpeak-NG
# def speak_text(text, language="en"):
#     os.system(f'espeak-ng -v {language} "{text}"')

# # Callback function to process audio in real time
# def callback(indata, frames, time, status):
#     if recognizer.AcceptWaveform(bytes(indata)):
#         result = json.loads(recognizer.Result())
#         recognized_text = result['text']
        
#         if recognized_text:
#             # Translate recognized Hindi text to English
#             tokenized_text = tokenizer.prepare_seq2seq_batch([recognized_text], return_tensors="pt")
#             translated = nmt_model.generate(**tokenized_text)
#             english_translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            
#             # Speak the translated text
#             speak_text(english_translation[0], language="en")

# def start_translation():
#     global audio_stream
#     audio_stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback)
#     audio_stream.start()

#     while isTranslateOn:
#         sd.sleep(100)  # Sleep to avoid high CPU usage

#     # Stop the audio stream when isTranslateOn becomes False
#     audio_stream.stop()
#     audio_stream.close()

# def stop_translation():
#     global isTranslateOn
#     isTranslateOn = False
#     output_placeholder.markdown("<h3 style='color:red;'>🚫 Translation Stopped</h3>", unsafe_allow_html=True)

# # UI layout
# st.title("🌐 Hindi to English Language Translator")

# # Instructions text
# st.markdown("""
#     <div style='background-color: black; padding: 10px; border-radius: 10px; color: white;'>
#         <h3>Instructions:</h3>
#         <p>1. Press "Start" to begin translation.</p>
#         <p>2. Press "Stop" to end the translation process.</p>
#     </div>
# """, unsafe_allow_html=True)

# # Buttons for Start and Stop
# col1, col2 = st.columns(2)
# with col1:
#     start_button = st.button("🟢 Start", use_container_width=True)
# with col2:
#     stop_button = st.button("🔴 Stop", use_container_width=True)

# # Output section
# output_placeholder = st.empty()

# # Check if "Start" button is clicked
# if start_button and not isTranslateOn:
#     isTranslateOn = True
#     output_placeholder.markdown("<h3 style='color:blue;'>🎙️ Listening... Speak now.</h3>", unsafe_allow_html=True)

#     # Start a new thread for the audio stream
#     threading.Thread(target=start_translation, daemon=True).start()

# # Check if "Stop" button is clicked
# if stop_button and isTranslateOn:
#     stop_translation()


import os
import json
import streamlit as st
import sounddevice as sd
import vosk
from transformers import MarianMTModel, MarianTokenizer
import threading

# Initialize pygame for audio playback
import pygame
pygame.mixer.init()

# Path to the Vosk Hindi model
model_path = r"D:\vigy-exp\Vigyaan-Translator-STS-main\vosk-model-small-hi-0.22"  # Change this to the correct path
model = vosk.Model(model_path)

# Initialize the recognizer with the model and sample rate (16000Hz)
recognizer = vosk.KaldiRecognizer(model, 16000)

# Specify the model for translating from Hindi to English
nmt_model_name = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = MarianTokenizer.from_pretrained(nmt_model_name)
nmt_model = MarianMTModel.from_pretrained(nmt_model_name)

# Flag to control the translation process
isTranslateOn = False
audio_stream = None

# Function for text-to-speech using eSpeak-NG
def speak_text(text, language="en"):
    os.system(f'espeak-ng -v {language} "{text}"')

# Callback function to process audio in real time
def callback(indata, frames, time, status):
    if not isTranslateOn:
        return  # Stop processing if translation is off

    if recognizer.AcceptWaveform(bytes(indata)):
        result = json.loads(recognizer.Result())
        recognized_text = result['text']
        
        if recognized_text:
            # Translate recognized Hindi text to English
            tokenized_text = tokenizer.prepare_seq2seq_batch([recognized_text], return_tensors="pt")
            translated = nmt_model.generate(**tokenized_text)
            english_translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            
            # Speak the translated text
            speak_text(english_translation[0], language="en")

def start_translation():
    global audio_stream
    audio_stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback)
    audio_stream.start()

    while isTranslateOn:
        sd.sleep(100)  # Sleep to avoid high CPU usage

    # Close the audio stream when isTranslateOn becomes False
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()

def stop_translation():
    global isTranslateOn, audio_stream
    isTranslateOn = False
    if audio_stream is not None:
        audio_stream.stop()  # Stop the stream immediately
        audio_stream.close()
        audio_stream = None  # Reset stream to None
    output_placeholder.markdown("<h3 style='color:red;'>🚫 Translation Stopped</h3>", unsafe_allow_html=True)

# UI layout
st.title("🌐 Hindi to English Language Translator")

# Instructions text
st.markdown("""
    <div style='background-color: black; padding: 10px; border-radius: 10px; color: white;'>
        <h3>Instructions:</h3>
        <p>1. Press "Start" to begin translation.</p>
        <p>2. Press "Stop" to end the translation process.</p>
    </div>
""", unsafe_allow_html=True)

# Buttons for Start and Stop
col1, col2 = st.columns(2)
with col1:
    start_button = st.button("🟢 Start", use_container_width=True)
with col2:
    stop_button = st.button("🔴 Stop", use_container_width=True)

# Output section
output_placeholder = st.empty()

# Check if "Start" button is clicked
if start_button and not isTranslateOn:
    isTranslateOn = True
    output_placeholder.markdown("<h3 style='color:blue;'>🎙️ Listening... Speak now.</h3>", unsafe_allow_html=True)

    # Start a new thread for the audio stream
    threading.Thread(target=start_translation, daemon=True).start()

# Check if "Stop" button is clicked
if stop_button and isTranslateOn:
    stop_translation()
