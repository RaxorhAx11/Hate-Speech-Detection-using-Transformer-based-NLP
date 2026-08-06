import React, { useState, useEffect } from "react";
import { useApp } from "../context/AppContext";
import { useSpeechSynthesis } from "../hooks/useSpeechSynthesis";
import { Server, Volume2, Info, Check, RefreshCw, AlertTriangle, Sliders } from "lucide-react";

export const Settings: React.FC = () => {
  const {
    apiUrl,
    updateApiUrl,
    isApiOnline,
    modelInfo,
    checkApiStatus,
    voiceVolume,
    updateVoiceVolume,
    voiceRate,
    updateVoiceRate,
    voiceURI,
    updateVoiceURI,
    maxWordLimit,
    updateMaxWordLimit,
    autoPredictOnStop,
    updateAutoPredictOnStop,
  } = useApp();

  const [inputUrl, setInputUrl] = useState(apiUrl);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [testing, setTesting] = useState(false);
  const [showApiConfig, setShowApiConfig] = useState(false);
  const { voices } = useSpeechSynthesis();

  // Synchronize internal input value with context apiUrl changes
  useEffect(() => {
    setInputUrl(apiUrl);
  }, [apiUrl]);

  const handleSaveUrl = async (e: React.FormEvent) => {
    e.preventDefault();
    setTesting(true);
    setSaveSuccess(false);

    try {
      updateApiUrl(inputUrl.trim());
      // The context will automatically trigger a status check when apiUrl changes
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err) {
      console.error(err);
    } finally {
      setTesting(false);
    }
  };

  const handleResetUrl = () => {
    setInputUrl("http://127.0.0.1:8000");
    updateApiUrl("http://127.0.0.1:8000");
  };

  return (
    <div className="max-w-4xl mx-auto space-y-12">
      {/* Page Header */}
      <div className="space-y-3">
        <span className="text-[11px] font-mono uppercase tracking-[0.28px] text-slate block">
          SYSTEM CONFIGURATION
        </span>
        <h1 className="text-4xl font-display font-light text-brand-primary tracking-[-0.48px]">
          Settings Panel
        </h1>
        <p className="text-sm text-body-muted leading-relaxed">
          Configure API connection endpoints, tune speech synthesis playback options, and audit model metadata details.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-start">
        
        {/* Left column: forms */}
        <div className="col-span-1 md:col-span-7 space-y-8">
          
          {/* Voice options settings */}
          <div className="bg-canvas border border-border-light rounded-sm p-6 space-y-6 shadow-sm">
            <div className="flex items-center gap-2 border-b border-hairline pb-3">
              <Volume2 className="h-4 w-4 text-brand-primary" />
              <h3 className="text-sm font-mono uppercase font-bold tracking-wider text-brand-primary">
                Voice Playback Options
              </h3>
            </div>

            <div className="space-y-5">
              {/* Voice Volume */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate uppercase">Synthesis Volume</span>
                  <span>{Math.round(voiceVolume * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={voiceVolume}
                  onChange={(e) => updateVoiceVolume(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-hairline rounded-lg appearance-none cursor-pointer accent-brand-primary"
                />
              </div>

              {/* Speech rate */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate uppercase">Speech Rate (Speed)</span>
                  <span>{voiceRate.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="2"
                  step="0.1"
                  value={voiceRate}
                  onChange={(e) => updateVoiceRate(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-hairline rounded-lg appearance-none cursor-pointer accent-brand-primary"
                />
              </div>

              {/* Browser voice selector */}
              {voices.length > 0 ? (
                <div className="space-y-1.5">
                  <label htmlFor="voice-select" className="text-xs font-mono text-slate uppercase">
                    Select Speech Voice
                  </label>
                  <select
                    id="voice-select"
                    value={voiceURI}
                    onChange={(e) => updateVoiceURI(e.target.value)}
                    className="w-full text-xs px-3 py-2.5 border border-border-light rounded-sm outline-none focus:border-form-focus transition-colors bg-canvas"
                  >
                    <option value="">Browser Default Voice</option>
                    {voices.map((v) => (
                      <option key={v.voiceURI} value={v.voiceURI}>
                        {v.name} ({v.lang})
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <p className="text-xs text-slate italic">
                  Local Speech Engine is loading or unsupported on this device.
                </p>
              )}
            </div>
          </div>

          {/* Text Input & Prediction Options */}
          <div className="bg-canvas border border-border-light rounded-sm p-6 space-y-6 shadow-sm">
            <div className="flex items-center gap-2 border-b border-hairline pb-3">
              <Sliders className="h-4 w-4 text-brand-primary" />
              <h3 className="text-sm font-mono uppercase font-bold tracking-wider text-brand-primary">
                Text & Prediction Options
              </h3>
            </div>

            <div className="space-y-5">
              {/* Max Word Limit */}
              <div className="space-y-1.5">
                <label htmlFor="max-word-limit" className="text-xs font-mono text-slate uppercase block font-semibold">
                  Max Input Word Limit
                </label>
                <div className="flex items-center gap-3">
                  <input
                    id="max-word-limit"
                    type="number"
                    min="100"
                    max="50000"
                    step="100"
                    value={maxWordLimit}
                    onChange={(e) => updateMaxWordLimit(Math.max(100, parseInt(e.target.value, 10) || 5000))}
                    className="w-32 text-xs px-3 py-2 border border-border-light rounded-sm outline-none focus:border-form-focus transition-colors bg-canvas"
                  />
                  <span className="text-xs text-slate italic">Words (at least 5000 is recommended)</span>
                </div>
              </div>

              {/* Auto-predict on Stop */}
              <div className="flex items-center gap-3 pt-2">
                <input
                  id="auto-predict-on-stop"
                  type="checkbox"
                  checked={autoPredictOnStop}
                  onChange={(e) => updateAutoPredictOnStop(e.target.checked)}
                  className="h-4 w-4 rounded border-border-light text-brand-primary focus:ring-focus-blue cursor-pointer"
                />
                <label htmlFor="auto-predict-on-stop" className="text-xs font-mono text-slate uppercase cursor-pointer select-none font-semibold">
                  Auto-predict after recording
                </label>
              </div>
            </div>
          </div>

          {/* Toggle for API connection settings */}
          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => setShowApiConfig(!showApiConfig)}
              className="text-xs font-mono text-slate hover:text-brand-primary underline transition-colors cursor-pointer"
            >
              {showApiConfig ? "Hide API Host Settings" : "Configure API Connection URL"}
            </button>
          </div>

          {/* API Server settings (visible only when toggled) */}
          {showApiConfig && (
            <div className="bg-canvas border border-border-light rounded-sm p-6 space-y-6 shadow-sm">
              <div className="flex items-center gap-2 border-b border-hairline pb-3">
                <Server className="h-4 w-4 text-brand-primary" />
                <h3 className="text-sm font-mono uppercase font-bold tracking-wider text-brand-primary">
                  FastAPI Host Settings
                </h3>
              </div>

              <form onSubmit={handleSaveUrl} className="space-y-4">
                <div className="space-y-1.5">
                  <label htmlFor="api-url" className="text-xs font-mono text-slate uppercase">
                    Service Gateway API URL
                  </label>
                  <input
                    id="api-url"
                    type="url"
                    value={inputUrl}
                    onChange={(e) => setInputUrl(e.target.value)}
                    placeholder="http://127.0.0.1:8000"
                    className="w-full text-xs px-4 py-3 border border-border-light rounded-sm outline-none focus:border-form-focus transition-colors"
                    required
                  />
                </div>

                <div className="flex items-center gap-3">
                  <button
                    type="submit"
                    disabled={testing}
                    className="inline-flex items-center justify-center rounded-full bg-brand-primary px-6 py-2.5 text-xs font-semibold text-white transition-all hover:bg-cohere-black active:scale-[0.98] disabled:opacity-50"
                  >
                    {testing ? "Connecting..." : "Save Configuration"}
                  </button>
                  <button
                    type="button"
                    onClick={handleResetUrl}
                    className="inline-flex items-center justify-center py-2 px-3 text-xs font-medium text-slate hover:text-brand-primary underline transition-colors font-mono"
                  >
                    Reset Default
                  </button>

                  {saveSuccess && (
                    <span className="text-xs font-mono text-deep-green flex items-center gap-1">
                      <Check className="h-3.5 w-3.5" />
                      Saved!
                    </span>
                  )}
                </div>
              </form>
            </div>
          )}

        </div>

        {/* Right column: read-only model metadata details */}
        <div className="col-span-1 md:col-span-5 space-y-6">
          
          <div className="bg-soft-stone border border-hairline/80 rounded-sm p-6 space-y-6">
            <div className="flex items-center gap-2 border-b border-hairline/60 pb-3">
              <Info className="h-4 w-4 text-brand-primary" />
              <h3 className="text-sm font-mono uppercase font-bold tracking-wider text-brand-primary">
                Model Information
              </h3>
            </div>

            {isApiOnline && modelInfo ? (
              <div className="space-y-4 text-xs font-mono">
                <div className="space-y-1">
                  <span className="text-slate uppercase block text-[10px]">Model Registry Tag</span>
                  <span className="font-semibold text-brand-primary break-all">{modelInfo.model_name}</span>
                </div>
                <div className="space-y-1">
                  <span className="text-slate uppercase block text-[10px]">Labels Count</span>
                  <span className="font-semibold text-brand-primary">{modelInfo.num_labels} classes</span>
                </div>
                <div className="space-y-1">
                  <span className="text-slate uppercase block text-[10px]">Max Sequence Length</span>
                  <span className="font-semibold text-brand-primary">{modelInfo.max_length} tokens</span>
                </div>
                <div className="space-y-1">
                  <span className="text-slate uppercase block text-[10px]">Dropout Coefficient</span>
                  <span className="font-semibold text-brand-primary">{modelInfo.dropout}</span>
                </div>
                <div className="space-y-1">
                  <span className="text-slate uppercase block text-[10px]">Active Execution Device</span>
                  <span className="font-semibold uppercase text-deep-green">{modelInfo.device}</span>
                </div>
                <div className="space-y-1">
                  <span className="text-slate uppercase block text-[10px]">Model Weight Path</span>
                  <span className="font-semibold text-brand-primary break-all">{modelInfo.model_path}</span>
                </div>
              </div>
            ) : (
              <div className="space-y-4 text-xs py-2 text-center text-slate">
                <AlertTriangle className="h-6 w-6 text-coral mx-auto mb-2" />
                <p>Unable to retrieve model information. The API server is currently unreachable.</p>
                <button
                  type="button"
                  onClick={checkApiStatus}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-hairline bg-canvas hover:bg-soft-stone rounded-sm font-medium transition-colors font-mono"
                >
                  <RefreshCw className="h-3 w-3" />
                  Retry Connection
                </button>
              </div>
            )}
          </div>

          <div className="bg-canvas border border-border-light rounded-sm p-6 text-xs text-body-muted leading-relaxed space-y-2">
            <span className="font-semibold text-brand-primary block font-mono">CONNECTION NOTE</span>
            <p>
              By default, predictions request from <code>http://127.0.0.1:8000</code>. Ensure that the python API is running via <code>python app.py</code> in the root directory before running the scan.
            </p>
          </div>

        </div>

      </div>
    </div>
  );
};
