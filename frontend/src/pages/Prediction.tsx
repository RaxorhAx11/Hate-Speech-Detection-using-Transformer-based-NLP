import React, { useState, useEffect } from "react";
import { useApp } from "../context/AppContext";
import { api } from "../api/client";
import type { PredictionResponse } from "../api/client";
import { useSpeechRecognition } from "../hooks/useSpeechRecognition";
import { useSpeechSynthesis } from "../hooks/useSpeechSynthesis";
import { Mic, MicOff, Volume2, Trash2, Clipboard, ArrowRight, Loader2, AlertCircle, RefreshCw } from "lucide-react";

export const Prediction: React.FC = () => {
  const { addToHistory } = useApp();
  
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  
  // Voice synthesis (Text to Speech)
  const { speak, isSpeaking, cancel: stopSpeaking } = useSpeechSynthesis();

  // Voice recording (Speech to Text)
  const [speechError, setSpeechError] = useState<string | null>(null);
  const [voiceStatus, setVoiceStatus] = useState<string>("Ready");
  
  const handleSpeechEnd = async (finalSpeechText: string) => {
    if (finalSpeechText.trim()) {
      setText(finalSpeechText);
      // Auto predict when speech ends
      await runPrediction(finalSpeechText);
    }
  };

  const {
    isListening,
    transcript,
    status: speechStatus,
    error: recognitionError,
    startListening,
    stopListening,
    retryListening,
    resetState: resetSpeechState,
    isSupported: isSpeechSupported,
  } = useSpeechRecognition({
    onEnd: handleSpeechEnd,
  });

  // Track character limit
  const maxCharLimit = 500;

  // Handle errors and status from speech recognition hook
  useEffect(() => {
    if (recognitionError) {
      setSpeechError(recognitionError);
    } else {
      setSpeechError(null);
    }
  }, [recognitionError]);

  useEffect(() => {
    if (speechStatus) {
      setVoiceStatus(speechStatus);
    }
  }, [speechStatus]);

  const handlePaste = async () => {
    try {
      const clipboardText = await navigator.clipboard.readText();
      if (clipboardText) {
        setText((prev) => (prev + " " + clipboardText).trim().substring(0, maxCharLimit));
        setError(null);
      }
    } catch (err) {
      console.warn("Could not access clipboard, falling back to manual paste:", err);
      setError("Please use Ctrl+V/Cmd+V to paste content.");
    }
  };

  const handleClear = () => {
    setText("");
    setResult(null);
    setError(null);
    setSpeechError(null);
    setVoiceStatus("Ready");
    stopSpeaking();
    resetSpeechState();
  };

  const handleStartRecording = () => {
    stopSpeaking();
    startListening();
  };

  const runPrediction = async (textToPredict = text) => {
    const trimmed = textToPredict.trim();
    if (!trimmed) {
      setError("Please write or speak some text to analyze.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setVoiceStatus("Processing...");
    stopSpeaking();

    try {
      const response = await api.predict(trimmed);
      setResult(response);
      
      // Save item to history
      addToHistory(
        trimmed,
        response.prediction,
        response.confidence,
        response.probabilities,
        response.processing_time_ms
      );

      // Auto TTS readout of prediction result
      const ttsMessage = `Prediction is ${response.prediction} with ${response.confidence.toFixed(1)} percent confidence.`;
      speak(ttsMessage);
      setVoiceStatus("Prediction Complete");

    } catch (err: any) {
      setError(err.message || "Failed to get prediction from server.");
      setVoiceStatus("Ready");
    } finally {
      setLoading(false);
    }
  };

  const speakResult = () => {
    if (result) {
      const label = result.prediction;
      const confidence = result.confidence;
      speak(`Predicted class is ${label} with confidence of ${confidence.toFixed(1)} percent.`);
    }
  };

  // Color mapping based on prediction label (minimal design)
  const getPredictionStyles = (label: string) => {
    switch (label) {
      case "Safe":
        return {
          bg: "bg-pale-green/10",
          border: "border-deep-green/30",
          text: "text-deep-green",
          badgeBg: "bg-pale-green",
        };
      case "Offensive":
        return {
          bg: "bg-coral-soft/5",
          border: "border-coral-soft/50",
          text: "text-coral",
          badgeBg: "bg-coral-soft/30",
        };
      case "Hate Speech":
        return {
          bg: "bg-error/5",
          border: "border-error/40",
          text: "text-error",
          badgeBg: "bg-error/20",
        };
      default:
        return {
          bg: "bg-soft-stone/20",
          border: "border-hairline",
          text: "text-brand-primary",
          badgeBg: "bg-soft-stone",
        };
    }
  };

  const resultStyle = result ? getPredictionStyles(result.prediction) : null;

  return (
    <div className="max-w-4xl mx-auto space-y-12">
      {/* Page Header */}
      <div className="space-y-3">
        <span className="text-[11px] font-mono uppercase tracking-[0.28px] text-slate block">
          WORKSPACE WORKBENCH
        </span>
        <h1 className="text-4xl font-display font-light text-brand-primary tracking-[-0.48px]">
          Analyze Text Commentary
        </h1>
        <p className="text-sm text-body-muted leading-relaxed">
          Input your comment or toggle the microphone to transcribe speech. Our model will return the toxicity classification breakdown instantly.
        </p>
      </div>

      {/* Main Form Section */}
      <div className="space-y-6">
        <div className="relative border border-border-light rounded-sm bg-canvas shadow-sm">
          
          {/* Text Area */}
          <textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value.substring(0, maxCharLimit));
              setError(null);
            }}
            placeholder="Write commentary here to analyze..."
            rows={5}
            className="w-full resize-none p-5 text-sm bg-transparent outline-none focus:border-form-focus transition-colors"
            style={{
              borderColor: error ? "var(--color-error)" : undefined,
            }}
            disabled={loading || isListening}
          />

          {/* Action Row inside Input */}
          <div className="flex items-center justify-between border-t border-border-light px-4 py-3 bg-soft-stone/10 text-xs">
            
            {/* Left buttons: Paste, Clear */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handlePaste}
                className="flex items-center gap-1 px-3 py-1.5 rounded-sm hover:bg-soft-stone text-brand-primary transition-colors focus-visible:ring-1 focus-visible:ring-focus-blue"
                title="Paste from clipboard"
              >
                <Clipboard className="h-3.5 w-3.5" />
                <span>Paste</span>
              </button>
              
              <button
                type="button"
                onClick={handleClear}
                className="flex items-center gap-1 px-3 py-1.5 rounded-sm hover:bg-error/15 text-slate hover:text-error transition-colors focus-visible:ring-1 focus-visible:ring-focus-blue"
                title="Clear input"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span>Clear</span>
              </button>
            </div>

            {/* Right: Char Counter */}
            <div className="text-slate font-mono">
              {text.length}/{maxCharLimit}
            </div>
          </div>
        </div>

        {/* Voice Assistant Panel */}
        {isSpeechSupported ? (
          <div className="bg-canvas border border-border-light rounded-sm p-6 space-y-6 shadow-sm">
            {/* Header Row */}
            <div className="flex items-center justify-between border-b border-hairline pb-3">
              <div className="flex items-center gap-2">
                <span className={`h-2.5 w-2.5 rounded-full ${isListening ? "bg-coral animate-pulse" : voiceStatus === "Processing..." ? "bg-action-blue animate-spin" : "bg-deep-green"}`} />
                <h3 className="text-xs font-mono uppercase font-bold tracking-wider text-brand-primary">
                  Voice Assistant
                </h3>
              </div>
              {/* Status Badge */}
              <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-mono font-semibold ${
                voiceStatus === "Listening..."
                  ? "bg-coral-soft/20 text-coral border border-coral/30"
                  : voiceStatus === "Processing..."
                  ? "bg-pale-blue text-action-blue border border-action-blue/30 animate-pulse"
                  : voiceStatus === "Prediction Complete"
                  ? "bg-pale-green text-deep-green border border-deep-green/30"
                  : voiceStatus.startsWith("Microphone") || voiceStatus.startsWith("Speech") || voiceStatus === "Network Error"
                  ? "bg-error/10 text-error border border-error/30"
                  : "bg-soft-stone text-brand-primary border border-hairline"
              }`}>
                {voiceStatus}
              </span>
            </div>
 
            {/* Transcript / Instructions */}
            <div className="space-y-2">
              {isListening ? (
                <div className="p-4 rounded-sm border border-coral-soft/40 bg-coral-soft/5 min-h-[60px] flex items-center justify-center text-center">
                  <p className="text-sm italic text-brand-primary">
                    {transcript ? `"${transcript}"` : "Microphone active. Speak now..."}
                  </p>
                </div>
              ) : (
                <div className="p-4 rounded-sm border border-hairline bg-soft-stone/10 min-h-[60px] flex items-center justify-center text-center text-xs text-body-muted leading-relaxed">
                  {voiceStatus === "Prediction Complete" && result ? (
                    <div className="space-y-1">
                      <p className="text-sm font-semibold text-brand-primary">
                        Transcribed commentary predicted as: <span className="underline">{result.prediction}</span>
                      </p>
                      <p className="italic">"{text}"</p>
                    </div>
                  ) : voiceStatus === "Microphone Permission Denied" ? (
                    <p className="text-error font-medium">
                      Microphone access was denied. Please check your browser site settings and allow microphone usage.
                    </p>
                  ) : voiceStatus === "No Speech Detected" ? (
                    <p className="text-coral font-medium">
                      No speech was detected. Press 'Start Recording' and try speaking again.
                    </p>
                  ) : voiceStatus === "Network Error" ? (
                    <p className="text-error font-medium">
                      Network connection to speech recognition servers failed. Please check your internet connection, disable any active VPN/firewall, or ensure the Web Speech API is enabled in your browser settings (e.g. Brave Shields).
                    </p>
                  ) : (
                    <p>
                      Click 'Start Recording' below, speak your commentary, and the system will automatically transcribe it, analyze for toxicity, and read out the result.
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Action Buttons Toolbar */}
            <div className="flex flex-wrap items-center gap-3">
              {!isListening ? (
                <button
                  type="button"
                  onClick={handleStartRecording}
                  disabled={loading || voiceStatus === "Processing..."}
                  className="inline-flex items-center justify-center gap-2 rounded-full bg-brand-primary px-5 py-2.5 text-xs font-semibold text-white transition-all hover:bg-cohere-black active:scale-[0.98] disabled:opacity-40"
                >
                  <Mic className="h-3.5 w-3.5" />
                  <span>Start Recording</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={stopListening}
                  className="inline-flex items-center justify-center gap-2 rounded-full bg-coral text-white border border-coral px-5 py-2.5 text-xs font-semibold transition-all hover:bg-coral/90 active:scale-[0.98]"
                >
                  <MicOff className="h-3.5 w-3.5" />
                  <span>Stop Recording</span>
                </button>
              )}

              {isListening && (
                <button
                  type="button"
                  onClick={retryListening}
                  className="inline-flex items-center justify-center gap-2 rounded-full border border-brand-primary text-brand-primary bg-transparent hover:bg-soft-stone px-4 py-2 text-xs font-medium transition-all"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  <span>Retry</span>
                </button>
              )}

              <button
                type="button"
                onClick={handleClear}
                disabled={loading || (!text && voiceStatus === "Ready")}
                className="inline-flex items-center justify-center gap-2 rounded-full border border-hairline text-slate hover:text-error hover:bg-error/5 px-4 py-2 text-xs font-medium transition-all disabled:opacity-40 disabled:hover:text-slate disabled:hover:bg-transparent"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span>Clear & Reset</span>
              </button>
            </div>
          </div>
        ) : (
          <div className="w-full text-xs text-slate border border-dashed border-hairline p-4 rounded-sm text-center bg-soft-stone/10">
            Speech synthesis or speech recognition is unavailable on this browser. Please use Chrome or Edge.
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="flex items-start gap-3 p-4 rounded-sm border border-error/30 bg-error/5 text-xs text-error">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            <div className="space-y-1">
              <span className="font-semibold">Inference Error:</span>
              <p className="leading-relaxed">{error}</p>
            </div>
          </div>
        )}

        {speechError && (
          <div className="flex items-start gap-3 p-4 rounded-sm border border-coral/30 bg-coral/5 text-xs text-coral">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            <div className="space-y-1">
              <span className="font-semibold">Microphone Status:</span>
              <p className="leading-relaxed">{speechError}</p>
            </div>
          </div>
        )}

        {/* Action Button Row */}
        <div className="flex flex-col sm:flex-row items-center gap-4">
          
          {/* Main Predict CTA */}
          <button
            onClick={() => runPrediction()}
            disabled={loading || isListening || !text.trim()}
            className="w-full sm:w-auto flex-1 inline-flex items-center justify-center gap-2 rounded-full bg-brand-primary px-8 py-3.5 text-sm font-semibold text-white transition-all hover:bg-cohere-black active:scale-[0.98] disabled:bg-slate/40 disabled:scale-100 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Running Transformer Inference...</span>
              </>
            ) : (
              <>
                <span>Analyze Commentary</span>
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Result Panel */}
      {result && resultStyle && (
        <section
          className={`rounded-sm p-6 md:p-8 border transition-all duration-300 ${resultStyle.bg} ${resultStyle.border}`}
        >
          <div className="flex flex-col md:flex-row gap-8 justify-between items-start md:items-center border-b border-hairline pb-6">
            
            {/* Title / Badge */}
            <div className="space-y-2">
              <span className="text-[10px] font-mono uppercase tracking-[0.28px] text-slate">
                PREDICTION OUTCOME
              </span>
              <div className="flex items-center gap-3">
                <span className={`text-3xl font-display font-light ${resultStyle.text}`}>
                  {result.prediction}
                </span>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold ${resultStyle.text} ${resultStyle.badgeBg}`}>
                  {result.confidence.toFixed(1)}% Confidence
                </span>
              </div>
            </div>

            {/* TTS playback button */}
            <button
              onClick={speakResult}
              disabled={isSpeaking}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-brand-primary text-brand-primary hover:bg-soft-stone text-xs font-medium focus-visible:ring-2 focus-visible:ring-focus-blue transition-all disabled:opacity-50"
            >
              <Volume2 className="h-3.5 w-3.5" />
              <span>{isSpeaking ? "Reading Aloud..." : "Read Prediction"}</span>
            </button>
          </div>

          {/* Probability distributions */}
          <div className="py-6 space-y-4">
            <h4 className="text-[10px] font-mono uppercase tracking-wider text-slate">
              Class Probability Distribution
            </h4>
            
            <div className="space-y-3">
              {/* Safe Progress */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span>Safe</span>
                  <span>{result.probabilities.Safe.toFixed(2)}%</span>
                </div>
                <div className="h-2 w-full bg-hairline rounded-full overflow-hidden">
                  <div
                    className="h-full bg-deep-green rounded-full transition-all duration-500"
                    style={{ width: `${result.probabilities.Safe}%` }}
                  />
                </div>
              </div>

              {/* Offensive Progress */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span>Offensive</span>
                  <span>{result.probabilities.Offensive.toFixed(2)}%</span>
                </div>
                <div className="h-2 w-full bg-hairline rounded-full overflow-hidden">
                  <div
                    className="h-full bg-coral rounded-full transition-all duration-500"
                    style={{ width: `${result.probabilities.Offensive}%` }}
                  />
                </div>
              </div>

              {/* Hate Speech Progress */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span>Hate Speech</span>
                  <span>{result.probabilities["Hate Speech"].toFixed(2)}%</span>
                </div>
                <div className="h-2 w-full bg-hairline rounded-full overflow-hidden">
                  <div
                    className="h-full bg-error rounded-full transition-all duration-500"
                    style={{ width: `${result.probabilities["Hate Speech"]}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Latency and technical metadata */}
          <div className="border-t border-hairline pt-4 flex justify-between items-center text-[11px] text-slate font-mono">
            <span>Inference Hardware: CPU</span>
            <span>API Latency: {result.processing_time_ms.toFixed(1)} ms</span>
          </div>

        </section>
      )}
    </div>
  );
};
