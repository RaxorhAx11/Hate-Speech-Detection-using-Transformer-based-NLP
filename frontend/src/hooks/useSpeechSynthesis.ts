import { useState, useEffect, useCallback } from "react";

export const useSpeechSynthesis = () => {
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Load available voices
  const loadVoices = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      const allVoices = window.speechSynthesis.getVoices();
      setVoices(allVoices);
    }
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      loadVoices();
      if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = loadVoices;
      }
    }
  }, [loadVoices]);

  const speak = useCallback((text: string) => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    // Cancel any ongoing speech first
    window.speechSynthesis.cancel();

    if (!text) return;

    const utterance = new SpeechSynthesisUtterance(text);

    // Load voice settings from localStorage
    const savedRate = localStorage.getItem("hate_speech_voice_rate");
    const savedVolume = localStorage.getItem("hate_speech_voice_volume");
    const savedVoiceURI = localStorage.getItem("hate_speech_voice_uri");

    utterance.rate = savedRate ? parseFloat(savedRate) : 1.0;
    utterance.volume = savedVolume ? parseFloat(savedVolume) : 1.0;

    if (savedVoiceURI) {
      const allVoices = window.speechSynthesis.getVoices();
      const selectedVoice = allVoices.find((v) => v.voiceURI === savedVoiceURI);
      if (selectedVoice) {
        utterance.voice = selectedVoice;
      }
    }

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = (e) => {
      console.error("Speech synthesis error:", e);
      setIsSpeaking(false);
    };

    window.speechSynthesis.speak(utterance);
  }, []);

  const cancel = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  }, []);

  return {
    voices,
    isSpeaking,
    speak,
    cancel,
    isSupported: typeof window !== "undefined" && !!window.speechSynthesis,
  };
};
