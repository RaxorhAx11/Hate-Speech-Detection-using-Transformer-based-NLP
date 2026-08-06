import { useState, useEffect, useRef } from "react";

interface UseSpeechRecognitionProps {
  onResult?: (text: string) => void;
  onEnd?: (text: string) => void;
}

export type SpeechStatus =
  | "Ready"
  | "Listening..."
  | "Processing..."
  | "Prediction Complete"
  | "Microphone Permission Denied"
  | "Speech Recognition Not Supported"
  | "No Speech Detected"
  | "Network Error";

export const useSpeechRecognition = (props?: UseSpeechRecognitionProps) => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [status, setStatus] = useState<SpeechStatus>("Ready");
  const [error, setError] = useState<string | null>(null);

  const recognitionRef = useRef<any>(null);
  const finalTranscriptRef = useRef("");
  const isListeningRef = useRef(false);

  // Keep references to callbacks updated to prevent effect re-runs
  const onResultRef = useRef(props?.onResult);
  const onEndRef = useRef(props?.onEnd);

  useEffect(() => {
    onResultRef.current = props?.onResult;
    onEndRef.current = props?.onEnd;
  }, [props?.onResult, props?.onEnd]);

  useEffect(() => {
    // Check compatibility
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setStatus("Speech Recognition Not Supported");
      setError("Speech recognition is not supported in this browser. Please use Google Chrome or Microsoft Edge.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false; // Single utterance
    recognition.interimResults = true; // interim transcript updates
    recognition.lang = "en-US";

    recognition.onstart = () => {
      setIsListening(true);
      isListeningRef.current = true;
      setStatus("Listening...");
      setError(null);
      setTranscript("");
      finalTranscriptRef.current = "";
    };

    recognition.onresult = (event: any) => {
      let interimTranscript = "";
      let finalTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      const activeTranscript = finalTranscript || interimTranscript;
      setTranscript(activeTranscript);

      if (finalTranscript) {
        finalTranscriptRef.current = finalTranscript;
        if (onResultRef.current) {
          onResultRef.current(finalTranscript);
        }
      }
    };

    recognition.onerror = (event: any) => {
      console.error("Speech recognition error event:", event.error);
      
      if (event.error === "no-speech") {
        setStatus("No Speech Detected");
        setError("No speech was detected. Please try speaking again.");
      } else if (event.error === "not-allowed") {
        setStatus("Microphone Permission Denied");
        setError("Microphone permission was denied. Please allow microphone access in settings.");
      } else if (event.error === "aborted") {
        // Aborted error can be triggered programmatically. Ignore displaying it as a scary error.
        console.warn("Speech recognition instance aborted.");
      } else if (event.error === "network") {
        setStatus("Network Error");
        setError("Network connection to speech recognition servers failed. Please check your internet connection, disable any active VPN/firewall, or ensure the Web Speech API is enabled in your browser settings (e.g. Brave Shields).");
      } else {
        setStatus("Ready");
        setError(`Speech recognition error: ${event.error}`);
      }
      setIsListening(false);
      isListeningRef.current = false;
    };

    recognition.onend = () => {
      setIsListening(false);
      isListeningRef.current = false;
      
      const finalVal = finalTranscriptRef.current.trim();
      if (finalVal) {
        setStatus("Processing...");
        if (onEndRef.current) {
          onEndRef.current(finalVal);
        }
      } else {
        // Reset status to Ready only if we didn't end up in an error status
        setStatus((prev) => {
          if (
            prev === "Microphone Permission Denied" ||
            prev === "Speech Recognition Not Supported" ||
            prev === "No Speech Detected" ||
            prev === "Network Error"
          ) {
            return prev;
          }
          return "Ready";
        });
      }
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch (e) {
          console.warn("Error during recognition abort cleanup:", e);
        }
      }
    };
  }, []); // Run once on mount

  const startListening = () => {
    if (!recognitionRef.current) {
      setError("Speech recognition is not initialized or supported.");
      return;
    }

    if (isListeningRef.current) {
      console.warn("Already listening, ignoring start command.");
      return;
    }

    try {
      setError(null);
      setStatus("Listening...");
      recognitionRef.current.start();
    } catch (e) {
      console.error("Failed to start speech recognition:", e);
      setError("Failed to start speech recognition.");
    }
  };

  const stopListening = () => {
    if (recognitionRef.current && isListeningRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        console.error("Failed to stop speech recognition:", e);
      }
    }
  };

  const cancelListening = () => {
    if (recognitionRef.current && isListeningRef.current) {
      try {
        recognitionRef.current.abort();
        setStatus("Ready");
        setError(null);
        setTranscript("");
        finalTranscriptRef.current = "";
      } catch (e) {
        console.error("Failed to abort speech recognition:", e);
      }
    }
  };

  const retryListening = () => {
    // Discard current and restart
    if (recognitionRef.current) {
      try {
        recognitionRef.current.abort();
      } catch (e) {
        console.warn("Abort during retry failed:", e);
      }
    }
    
    // Tiny delay to ensure previous instance is cleaned up by the browser event loop
    setTimeout(() => {
      startListening();
    }, 100);
  };

  const resetState = () => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.abort();
      } catch (e) {
        console.warn("Abort during reset failed:", e);
      }
    }
    setTranscript("");
    finalTranscriptRef.current = "";
    setError(null);
    setStatus("Ready");
  };

  return {
    isListening,
    transcript,
    status,
    error,
    startListening,
    stopListening,
    cancelListening,
    retryListening,
    resetState,
    isSupported: !!((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition),
  };
};
