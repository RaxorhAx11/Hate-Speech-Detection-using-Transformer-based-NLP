export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string | null;
  model_name: string;
  timestamp: number;
}

export interface ModelInfoResponse {
  model_name: string;
  num_labels: number;
  max_length: number;
  dropout: number;
  labels: string[];
  device: string;
  model_path: string;
  model_loaded: boolean;
}

export interface Probabilities {
  Safe: number;
  Offensive: number;
  "Hate Speech": number;
}

export interface PredictionResponse {
  prediction: string;
  confidence: number;
  probabilities: Probabilities;
  processing_time_ms: number;
}

export interface ClassMetricItem {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface MetricsResponse {
  available: boolean;
  error?: string;
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  roc_auc_macro: number;
  class_metrics: {
    Safe?: ClassMetricItem;
    Offensive?: ClassMetricItem;
    "Hate Speech"?: ClassMetricItem;
  };
}

// Fetch API base URL from settings (localStorage) or fall back to default
export const getApiBaseUrl = (): string => {
  const savedUrl = localStorage.getItem("hate_speech_api_url");
  if (savedUrl) {
    return savedUrl.replace(/\/$/, ""); // Strip trailing slash
  }
  return "http://127.0.0.1:8000";
};

// Helper for fetching with a timeout
const fetchWithTimeout = async (
  url: string,
  options: RequestInit = {},
  timeoutMs = 8000
): Promise<Response> => {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (error: any) {
    if (error.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs}ms`);
    }
    throw error;
  } finally {
    clearTimeout(id);
  }
};

export const api = {
  async getHealth(): Promise<HealthResponse> {
    const baseUrl = getApiBaseUrl();
    try {
      const response = await fetchWithTimeout(`${baseUrl}/health`, { method: "GET" }, 4000);
      if (!response.ok) {
        throw new Error(`API responded with status code ${response.status}`);
      }
      return await response.json();
    } catch (error: any) {
      console.error("Health check failed:", error);
      throw new Error(error.message || "Failed to reach the API server. Please check if the backend is running.");
    }
  },

  async getModelInfo(): Promise<ModelInfoResponse> {
    const baseUrl = getApiBaseUrl();
    try {
      const response = await fetchWithTimeout(`${baseUrl}/model-info`, { method: "GET" }, 4000);
      if (!response.ok) {
        throw new Error(`API responded with status code ${response.status}`);
      }
      return await response.json();
    } catch (error: any) {
      console.error("Model info fetch failed:", error);
      throw new Error(error.message || "Failed to fetch model metadata.");
    }
  },

  async predict(text: string): Promise<PredictionResponse> {
    const baseUrl = getApiBaseUrl();
    try {
      const response = await fetchWithTimeout(
        `${baseUrl}/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text }),
        },
        10000 // 10s timeout for models running inference
      );

      if (response.status === 422) {
        const errData = await response.json();
        throw new Error(errData.detail || "Validation Error: Input text is invalid.");
      }

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`API prediction error (${response.status}): ${errText || "Unknown error"}`);
      }

      const data = await response.json();
      
      const probs = data.probabilities || {};
      
      const mappedProbs: Probabilities = {
        Safe: typeof probs.Safe === "number" ? probs.Safe : 0,
        Offensive: typeof probs.Offensive === "number" ? probs.Offensive : 0,
        "Hate Speech": typeof probs["Hate Speech"] === "number" ? probs["Hate Speech"] : 0,
      };

      return {
        prediction: data.prediction || "Unknown",
        confidence: typeof data.confidence === "number" ? data.confidence : 0,
        probabilities: mappedProbs,
        processing_time_ms: typeof data.processing_time_ms === "number" ? data.processing_time_ms : 0,
      };
    } catch (error: any) {
      console.error("Prediction request failed:", error);
      throw new Error(error.message || "Failed to communicate with prediction API.");
    }
  },

  async getMetrics(): Promise<MetricsResponse> {
    const baseUrl = getApiBaseUrl();
    try {
      const response = await fetchWithTimeout(`${baseUrl}/metrics`, { method: "GET" }, 4000);
      if (!response.ok) {
        throw new Error(`API responded with status code ${response.status}`);
      }
      return await response.json();
    } catch (error: any) {
      console.error("Metrics fetch failed:", error);
      throw new Error(error.message || "Failed to fetch model evaluation metrics.");
    }
  },
};
