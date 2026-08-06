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

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.onstart = null;
          recognitionRef.current.onresult = null;
          recognitionRef.current.onerror = null;
          recognitionRef.current.onend = null;
          recognitionRef.current.abort();
        } catch (e) {
          console.warn("Error during recognition abort cleanup on unmount:", e);
        }
      }
    };
  }, []);

  const startListening = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setStatus("Speech Recognition Not Supported");
      setError("Speech recognition is not supported in this browser. Please use Google Chrome or Microsoft Edge.");
      return;
    }

    // Stop and clean up any existing instance first
    if (recognitionRef.current) {
      try {
        recognitionRef.current.onstart = null;
        recognitionRef.current.onresult = null;
        recognitionRef.current.onerror = null;
        recognitionRef.current.onend = null;
        recognitionRef.current.abort();
      } catch (e) {
        console.warn("Error aborting previous speech recognition:", e);
      }
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = true; // KEEP LISTENING CONTINUOUSLY UNTIL STOPPED
      recognition.interimResults = true; // Interim updates
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

        // Iterate over all results in the current session
        for (let i = 0; i < event.results.length; ++i) {
          const result = event.results[i];
          if (result.isFinal) {
            finalTranscript += result[0].transcript;
          } else {
            interimTranscript += result[0].transcript;
          }
        }

        const activeTranscript = finalTranscript + interimTranscript;
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
        
        let errorStatus: SpeechStatus = "Ready";
        let errorMessage = "";

        if (event.error === "no-speech") {
          errorStatus = "No Speech Detected";
          errorMessage = "No speech was detected. Please try speaking again.";
        } else if (event.error === "not-allowed") {
          errorStatus = "Microphone Permission Denied";
          errorMessage = "Microphone permission was denied. Please allow microphone access in browser settings.";
        } else if (event.error === "network") {
          errorStatus = "Network Error";
          errorMessage = "Network connection to speech recognition servers failed. Please check your internet connection or Brave shields.";
        } else if (event.error === "aborted") {
          console.warn("Speech recognition instance aborted.");
          return; // Do not show scary errors for programmatic stop/abort
        } else {
          errorMessage = `Speech recognition error: ${event.error}`;
        }

        if (errorMessage) {
          setStatus(errorStatus);
          setError(errorMessage);
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
      recognition.start();
    } catch (e: any) {
      console.error("Failed to start speech recognition:", e);
      setError(`Failed to start speech recognition: ${e.message || e}`);
      setStatus("Ready");
      setIsListening(false);
      isListeningRef.current = false;
    }
  };

  const stopListening = () => {
    if (recognitionRef.current && isListeningRef.current) {
      try {
        // stop() will stop active recording and trigger onend, which handles predicting
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
    startListening();
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
    isSupported: typeof window !== "undefined" && !!((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition),
  };
};
