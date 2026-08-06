import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { api } from "../api/client";
import type { HealthResponse, ModelInfoResponse, Probabilities, MetricsResponse } from "../api/client";

export type TabType = "home" | "prediction" | "history" | "settings" | "about";

export interface HistoryItem {
  id: string;
  text: string;
  prediction: string;
  confidence: number;
  probabilities: Probabilities;
  processingTimeMs: number;
  timestamp: string;
}

interface AppContextProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  apiUrl: string;
  updateApiUrl: (url: string) => void;
  isApiOnline: boolean | null; // null = checking, true = online, false = offline
  apiMetadata: HealthResponse | null;
  modelInfo: ModelInfoResponse | null;
  metrics: MetricsResponse | null;
  checkApiStatus: () => Promise<void>;
  history: HistoryItem[];
  addToHistory: (text: string, prediction: string, confidence: number, probabilities: Probabilities, processingTime: number) => void;
  deleteFromHistory: (id: string) => void;
  clearHistory: () => void;
  voiceVolume: number;
  updateVoiceVolume: (volume: number) => void;
  voiceRate: number;
  updateVoiceRate: (rate: number) => void;
  voiceURI: string;
  updateVoiceURI: (uri: string) => void;
}

const AppContext = createContext<AppContextProps | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activeTab, setActiveTab] = useState<TabType>("home");
  
  // Settings & Configuration States
  const [apiUrl, setApiUrl] = useState(() => {
    return localStorage.getItem("hate_speech_api_url") || "http://127.0.0.1:8000";
  });
  
  const [voiceVolume, setVoiceVolume] = useState(() => {
    const saved = localStorage.getItem("hate_speech_voice_volume");
    return saved ? parseFloat(saved) : 1.0;
  });

  const [voiceRate, setVoiceRate] = useState(() => {
    const saved = localStorage.getItem("hate_speech_voice_rate");
    return saved ? parseFloat(saved) : 1.0;
  });

  const [voiceURI, setVoiceURI] = useState(() => {
    return localStorage.getItem("hate_speech_voice_uri") || "";
  });

  // History State
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    const saved = localStorage.getItem("hate_speech_history");
    return saved ? JSON.parse(saved) : [];
  });

  // API Status States
  const [isApiOnline, setIsApiOnline] = useState<boolean | null>(null);
  const [apiMetadata, setApiMetadata] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  // Sync settings helper
  const updateApiUrl = (url: string) => {
    localStorage.setItem("hate_speech_api_url", url);
    setApiUrl(url);
  };

  const updateVoiceVolume = (volume: number) => {
    localStorage.setItem("hate_speech_voice_volume", volume.toString());
    setVoiceVolume(volume);
  };

  const updateVoiceRate = (rate: number) => {
    localStorage.setItem("hate_speech_voice_rate", rate.toString());
    setVoiceRate(rate);
  };

  const updateVoiceURI = (uri: string) => {
    localStorage.setItem("hate_speech_voice_uri", uri);
    setVoiceURI(uri);
  };

  // Perform API checking
  const checkApiStatus = useCallback(async () => {
    setIsApiOnline(null);
    try {
      const health = await api.getHealth();
      setIsApiOnline(true);
      setApiMetadata(health);

      // Fetch model info
      try {
        const info = await api.getModelInfo();
        setModelInfo(info);
      } catch (e) {
        console.warn("Could not retrieve model metadata:", e);
      }

      // Fetch metrics
      try {
        const met = await api.getMetrics();
        setMetrics(met);
      } catch (e) {
        console.warn("Could not retrieve model metrics:", e);
      }
    } catch (error) {
      console.warn("API status check failed:", error);
      setIsApiOnline(false);
      setApiMetadata(null);
      setModelInfo(null);
      setMetrics(null);
    }
  }, []);

  // Run status check on mount and whenever the API url changes
  useEffect(() => {
    checkApiStatus();
  }, [apiUrl, checkApiStatus]);

  // Sync history state to local storage
  useEffect(() => {
    localStorage.setItem("hate_speech_history", JSON.stringify(history));
  }, [history]);

  const addToHistory = (
    text: string,
    prediction: string,
    confidence: number,
    probabilities: Probabilities,
    processingTimeMs: number
  ) => {
    const newItem: HistoryItem = {
      id: crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2, 9),
      text,
      prediction,
      confidence,
      probabilities,
      processingTimeMs,
      timestamp: new Date().toISOString(),
    };
    setHistory((prev) => [newItem, ...prev]);
  };

  const deleteFromHistory = (id: string) => {
    setHistory((prev) => prev.filter((item) => item.id !== id));
  };

  const clearHistory = () => {
    setHistory([]);
  };

  return (
    <AppContext.Provider
      value={{
        activeTab,
        setActiveTab,
        apiUrl,
        updateApiUrl,
        isApiOnline,
        apiMetadata,
        modelInfo,
        metrics,
        checkApiStatus,
        history,
        addToHistory,
        deleteFromHistory,
        clearHistory,
        voiceVolume,
        updateVoiceVolume,
        voiceRate,
        updateVoiceRate,
        voiceURI,
        updateVoiceURI,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useApp must be used within an AppProvider");
  }
  return context;
};
