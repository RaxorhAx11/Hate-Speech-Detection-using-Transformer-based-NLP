import os
import logging
import time
import speech_recognition as sr
import pyttsx3
from inference import HateSpeechInference
from config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VoiceInterface:
    def __init__(self, model_path: str = None):
        self.config = load_config()
        self.inference_engine = None
        
        # Determine model path
        self.model_path = model_path or os.path.join(self.config.training.output_dir, "best_model")
        
        # Initialize text-to-speech engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", self.config.voice.speech_rate)
            self.tts_engine.setProperty("volume", self.config.voice.volume)
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3 TTS engine: {e}. Voice output will be printed only.")
            self.tts_engine = None
            
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()

    def load_model(self):
        """Loads inference model if not already loaded."""
        if self.inference_engine is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. Please run train.py to train the model first."
                )
            self.inference_engine = HateSpeechInference(model_path=self.model_path)

    def speak(self, text: str):
        """Speaks the text using TTS and prints it."""
        print(f"[Speech Output]: {text}")
        if self.tts_engine:
            try:
                # pyttsx3 can sometimes hang if runAndWait is not cleaned up
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS speaking failed: {e}")

    def record_and_recognize(self) -> str:
        """Records voice from the microphone and converts it to text."""
        print("\n--- VOICE INPUT ---")
        
        # Try standard speech_recognition Microphone (which requires PyAudio)
        try:
            with sr.Microphone() as source:
                print("Adjusting for ambient noise... Please wait.")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.speak("I am listening. Please speak now...")
                print("Listening (Speak for up to 5 seconds)...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
            print("Processing voice...")
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized: \"{text}\"")
            return text
        except (sr.RequestError, sr.UnknownValueError) as e:
            logger.warning(f"Google speech recognition failed: {e}")
            self.speak("I could not understand the speech. Please try again.")
            return ""
        except Exception as microphone_error:
            # Fallback to sounddevice + soundfile recording if PyAudio is missing/errors out
            logger.warning(f"Standard microphone failed: {microphone_error}. Trying sounddevice fallback.")
            try:
                import sounddevice as sd
                import soundfile as sf
                
                fs = 16000
                duration = 5 # seconds
                temp_filename = "temp_recording.wav"
                
                self.speak(f"Recording from default input for {duration} seconds. Start speaking now...")
                print("Recording...")
                recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
                sd.wait()
                print("Finished recording. Processing...")
                sf.write(temp_filename, recording, fs)
                
                # Recognize from the saved file
                with sr.AudioFile(temp_filename) as source:
                    audio_data = self.recognizer.record(source)
                
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

    def run_prediction(self, text: str):
        """Runs the classifier and speaks the class prediction and confidence."""
        if not text.strip():
            self.speak("No input text was recognized.")
            return
            
        try:
            self.load_model()
            result = self.inference_engine.predict(text)
            prediction = result["prediction"]
            confidence = result["confidence"]
            
            # Formulate speech output
            speech_out = f"Predicted category is {prediction} with a confidence of {confidence:.2f} percent."
            self.speak(speech_out)
            
            # Print details
            print(f"Probabilities: {result['probabilities']}")
        except Exception as e:
            logger.error(f"Inference run failed: {e}")
            self.speak("An error occurred while running prediction on your speech.")

    def run_loop(self):
        """Interactive loop allowing the user to select text or voice inputs."""
        print("="*50)
        print("  HATE SPEECH DETECTION VOICE & TEXT CONSOLE  ")
        print("="*50)
        
        # Pre-check model availability
        if not os.path.exists(self.model_path):
            print(f"[Warning] Best model not found at {self.model_path}.")
            print("Please make sure you train the model first by running 'python train.py'.")
            
        while True:
            print("\nSelect mode:")
            print("1. Text input prediction")
            print("2. Voice input prediction")
            print("3. Exit")
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == "1":
                text = input("Enter text: ").strip()
                if text:
                    self.run_prediction(text)
            elif choice == "2":
                recognized_text = self.record_and_recognize()
                if recognized_text:
                    self.run_prediction(recognized_text)
            elif choice == "3":
                print("Exiting console.")
                break
            else:
                print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    interface = VoiceInterface()
    interface.run_loop()
