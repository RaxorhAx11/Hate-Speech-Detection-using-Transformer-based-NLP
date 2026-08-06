import os
import logging
import time
import requests
import speech_recognition as sr
import pyttsx3
from config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VoiceInterface:
    def __init__(self, model_path: str = None, api_url: str = None):
        self.config = load_config()
        self.inference_engine = None
        
        # Determine paths and endpoints
        self.model_path = model_path or os.path.join(self.config.training.output_dir, "best_model")
        self.api_url = api_url or self.config.voice.api_url or "http://127.0.0.1:8000"
        
        # Initialize text-to-speech engine
        self._init_tts()
            
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()

    def _init_tts(self):
        """Initializes or re-initializes the TTS engine."""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", self.config.voice.speech_rate)
            self.tts_engine.setProperty("volume", self.config.voice.volume)
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3 TTS engine: {e}. Voice output will be printed only.")
            self.tts_engine = None

    def speak(self, text: str):
        """Speaks the text using TTS and prints it."""
        print(f"[Speech Output]: {text}")
        if self.tts_engine:
            try:
                # Re-initialize to avoid potential thread lockups in pyttsx3
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS speaking failed: {e}. Re-initializing engine.")
                self._init_tts()
                # Attempt once more after re-init
                try:
                    if self.tts_engine:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                except Exception as ex:
                    logger.error(f"TTS retry failed: {ex}")

    def record_and_recognize(self, source=None) -> str:
        """Records voice from the microphone and converts it to text."""
        # If no active continuous source is provided, open a single-use mic
        try:
            if source is None:
                with sr.Microphone() as mic_source:
                    print("Adjusting for ambient noise... Please wait.")
                    self.recognizer.adjust_for_ambient_noise(mic_source, duration=1)
                    self.speak("I am listening. Please speak now...")
                    print("Listening (Speak for up to 5 seconds)...")
                    audio = self.recognizer.listen(mic_source, timeout=5, phrase_time_limit=5)
            else:
                # Reuse the open continuous microphone source
                print("\nListening...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)

            print("Processing voice...")
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized: \"{text}\"")
            return text
        except sr.WaitTimeoutError:
            # Silent timeout is expected in continuous mode
            if source is None:
                print("Listening timed out. No speech detected.")
            return ""
        except (sr.RequestError, sr.UnknownValueError) as e:
            logger.warning(f"Google speech recognition failed: {e}")
            self.speak("I could not understand the speech. Please try again.")
            return ""
        except Exception as microphone_error:
            # Fallback to sounddevice + soundfile recording if PyAudio is missing
            logger.warning(f"Standard microphone failed: {microphone_error}. Trying sounddevice fallback.")
            return self._record_sounddevice_fallback()

    def _record_sounddevice_fallback(self) -> str:
        """Fallback recording using sounddevice when PyAudio/SpeechRecognition mic is unavailable."""
        try:
            import sounddevice as sd
            import soundfile as sf
            
            fs = 16000
            duration = 5
            temp_filename = "temp_recording.wav"
            
            self.speak(f"Recording from default input for {duration} seconds. Start speaking now...")
            print("Recording...")
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
            sd.wait()
            print("Finished recording. Processing...")
            sf.write(temp_filename, recording, fs)
            
            # Recognize from the saved file
            with sr.AudioFile(temp_filename) as file_source:
                audio_data = self.recognizer.record(file_source)
            
            # Clean up temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
            text = self.recognizer.recognize_google(audio_data)
            print(f"Recognized: \"{text}\"")
            return text
        except Exception as fallback_error:
            logger.error(f"Voice fallback failed: {fallback_error}")
            self.speak("Could not access microphone or record audio.")
            return ""

    def run_prediction(self, text: str, use_local_fallback: bool = True):
        """Sends recognized text to the FastAPI prediction endpoint, with local inference fallback."""
        if not text.strip():
            self.speak("No input text was recognized.")
            return
            
        try:
            # Send text to API
            print(f"Sending request to API: {self.api_url}/predict")
            response = requests.post(
                f"{self.api_url}/predict",
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                confidence = result["confidence"]
                
                speech_out = f"Predicted category is {prediction} with a confidence of {confidence:.2f} percent."
                self.speak(speech_out)
                print(f"Probabilities: {result['probabilities']}")
                print(f"API processing time: {result['processing_time_ms']} ms")
            else:
                print(f"API error response (Status: {response.status_code}): {response.text}")
                self.speak("The prediction server returned an error.")
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_err:
            logger.warning(f"Connection to prediction API failed: {conn_err}")
            
            if use_local_fallback:
                print("Running local fallback prediction...")
                self._run_local_prediction(text)
            else:
                self.speak("Failed to connect to the prediction API.")

    def _run_local_prediction(self, text: str):
        """Initializes and runs prediction locally if API is unreachable."""
        try:
            if self.inference_engine is None:
                # Lazy load local model
                from inference import HateSpeechInference
                if not os.path.exists(self.model_path):
                    self.speak("API is unreachable and local model files were not found.")
                    return
                self.speak("Loading local model files for fallback inference...")
                self.inference_engine = HateSpeechInference(model_path=self.model_path)
            
            result = self.inference_engine.predict(text, explain=False)
            prediction = result["prediction"]
            confidence = result["confidence"]
            
            speech_out = f"Local Fallback: Predicted category is {prediction} with a confidence of {confidence:.2f} percent."
            self.speak(speech_out)
            print(f"Probabilities: {result['probabilities']}")
            
        except Exception as local_err:
            logger.error(f"Local inference fallback failed: {local_err}")
            self.speak("An error occurred during local prediction fallback.")

    def run_continuous(self):
        """Continuous voice recognition mode listening in a loop until a exit command is spoken."""
        print("\n" + "="*50)
        print("     CONTINUOUS VOICE RECOGNITION MODE ACTIVE     ")
        print("Say 'stop continuous', 'stop loop', or 'exit' to quit.")
        print("="*50)
        
        try:
            with sr.Microphone() as mic_source:
                print("Adjusting for ambient noise... Please wait.")
                self.recognizer.adjust_for_ambient_noise(mic_source, duration=1.5)
                self.speak("Continuous listening is active. Speak when ready...")
                
                while True:
                    text = self.record_and_recognize(source=mic_source)
                    if text:
                        # Clean trigger words
                        clean_text = text.lower().strip()
                        if clean_text in ["stop continuous", "stop loop", "exit", "quit", "stop listening"]:
                            self.speak("Exiting continuous voice mode.")
                            break
                        
                        self.run_prediction(text)
                        # Small pause before listening again
                        time.sleep(1)
        except Exception as e:
            logger.error(f"Continuous voice loop failed: {e}")
            self.speak("Could not start continuous microphone source. Check your PyAudio installation.")

    def run_loop(self):
        """Interactive console menu selection."""
        print("="*50)
        print("  HATE SPEECH DETECTION VOICE & TEXT CONSOLE  ")
        print("="*50)
        
        while True:
            print("\nSelect mode:")
            print("1. Text input prediction")
            print("2. Voice input prediction (Single phrase)")
            print("3. Continuous voice recognition loop")
            print("4. Exit")
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == "1":
                text = input("Enter text: ").strip()
                if text:
                    self.run_prediction(text)
            elif choice == "2":
                recognized_text = self.record_and_recognize()
                if recognized_text:
                    self.run_prediction(recognized_text)
            elif choice == "3":
                self.run_continuous()
            elif choice == "4":
                print("Exiting console.")
                break
            else:
                print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    interface = VoiceInterface()
    interface.run_loop()
