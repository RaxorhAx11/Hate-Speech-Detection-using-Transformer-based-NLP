import React from "react";
import { useApp } from "../context/AppContext";
import { ArrowRight, Shield, Award, Cpu, Timer } from "lucide-react";

export const Home: React.FC = () => {
  const { setActiveTab, modelInfo, isApiOnline, metrics, history } = useApp();

  const f1ScoreDisplay = metrics?.available && metrics?.f1_macro
    ? `${(metrics.f1_macro * 100).toFixed(1)}%`
    : "84.3%";

  const latencyDisplay = React.useMemo(() => {
    if (!history || history.length === 0) return "~12ms";
    const total = history.reduce((sum, item) => sum + item.processingTimeMs, 0);
    return `${(total / history.length).toFixed(1)}ms`;
  }, [history]);

  return (
    <div className="space-y-20 md:space-y-28">
      {/* Hero Section */}
      <section className="text-center max-w-4xl mx-auto space-y-8">
        <span className="text-coral text-[12px] font-mono uppercase tracking-[0.28px] font-semibold border border-coral-soft/30 px-3 py-1 rounded-full bg-coral-soft/5">
          Transformer-Powered NLP Model
        </span>
        
        {/* Monumental Display Headline (from DESIGN.md typography hero-display) */}
        <h1 className="text-5xl md:text-7xl font-display font-light text-brand-primary leading-[1.0] tracking-[-1.92px]">
          Hate Speech Detection using Transformer NLP.
        </h1>
        
        <p className="text-base md:text-lg text-body-muted font-body leading-relaxed max-w-2xl mx-auto">
          Analyze online commentary in real-time. Detect Safe, Offensive, and targeted Hate Speech with token-level confidence scores, classification details, and voice playback.
        </p>

        {/* Buttons (from DESIGN.md components) */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
          <button
            onClick={() => setActiveTab("prediction")}
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-full bg-brand-primary px-8 py-3.5 text-sm font-medium text-white transition-all hover:bg-cohere-black active:scale-[0.98]"
          >
            Start Analyzing Text
            <ArrowRight className="h-4 w-4" />
          </button>
          
          <button
            onClick={() => setActiveTab("about")}
            className="w-full sm:w-auto inline-flex items-center justify-center py-2.5 text-sm font-medium text-brand-primary underline underline-offset-4 hover:text-action-blue transition-colors"
          >
            Read Technical Specifications
          </button>
        </div>
      </section>

      {/* Trust & Performance Metrics Banner */}
      <section className="grid grid-cols-1 md:grid-cols-4 gap-6 border-y border-hairline/60 py-10">
        <div className="flex flex-col items-center md:items-start text-center md:text-left p-4">
          <Award className="h-5 w-5 text-deep-green mb-3" />
          <h3 className="font-mono text-xs uppercase tracking-wider text-slate mb-1">Target F1-Score</h3>
          <span className="text-2xl font-light text-brand-primary">{f1ScoreDisplay}</span>
        </div>
        <div className="flex flex-col items-center md:items-start text-center md:text-left p-4">
          <Timer className="h-5 w-5 text-deep-green mb-3" />
          <h3 className="font-mono text-xs uppercase tracking-wider text-slate mb-1">Average Latency</h3>
          <span className="text-2xl font-light text-brand-primary">{latencyDisplay}</span>
        </div>
        <div className="flex flex-col items-center md:items-start text-center md:text-left p-4">
          <Cpu className="h-5 w-5 text-deep-green mb-3" />
          <h3 className="font-mono text-xs uppercase tracking-wider text-slate mb-1">Active Backend Model</h3>
          <span className="text-2xl font-light text-brand-primary truncate max-w-full">
            {isApiOnline && modelInfo ? modelInfo.model_name.split("/").pop() : "DistilBERT"}
          </span>
        </div>
        <div className="flex flex-col items-center md:items-start text-center md:text-left p-4">
          <Shield className="h-5 w-5 text-deep-green mb-3" />
          <h3 className="font-mono text-xs uppercase tracking-wider text-slate mb-1">Unified Datasets</h3>
          <span className="text-2xl font-light text-brand-primary">5 Sources</span>
        </div>
      </section>

      {/* Model Information Card (DESIGN.md product-card style) */}
      <section className="bg-soft-stone rounded-sm p-8 md:p-12 max-w-5xl mx-auto flex flex-col md:flex-row gap-8 items-center">
        <div className="flex-1 space-y-4">
          <span className="text-xs font-mono tracking-wider text-slate uppercase block">
            AI ARCHITECTURE OVERVIEW
          </span>
          <h2 className="text-3xl font-display font-light text-brand-primary leading-tight">
            Fine-tuned for balanced classification.
          </h2>
          <p className="text-sm text-body-muted leading-relaxed">
            The model is fine-tuned on custom class weights to balance representation across Safe, Offensive, and Hate Speech annotations. The deployment is hosted in a FastAPI web service, utilizing attention mask configurations to secure deterministic, high-speed predictions.
          </p>
          <div className="flex gap-4 pt-2">
            <div className="text-xs font-mono border border-hairline px-2.5 py-1 rounded bg-canvas/60 text-brand-primary">
              MAX LENGTH: {modelInfo?.max_length || 128}
            </div>
            <div className="text-xs font-mono border border-hairline px-2.5 py-1 rounded bg-canvas/60 text-brand-primary">
              CLASSES: Safe, Offensive, Hate Speech
            </div>
          </div>
        </div>
        
        {/* Simple mock response panel (DESIGN.md agent-console-card style) */}
        <div className="w-full md:w-[360px] bg-brand-primary text-white rounded-sm p-6 space-y-4 font-mono text-xs self-stretch flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-2">
              <span className="text-slate">SAMPLE INFERENCE STATUS</span>
              <span className="text-deep-green animate-pulse">●</span>
            </div>
            <div className="bg-white/5 p-3 rounded-sm text-slate border border-white/5">
              "We must support equality and defend civil rights for all."
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span>PREDICTION:</span>
              <span className="text-pale-green font-semibold uppercase">SAFE</span>
            </div>
            <div className="flex items-center justify-between">
              <span>CONFIDENCE:</span>
              <span>99.82%</span>
            </div>
            <div className="flex items-center justify-between">
              <span>LATENCY:</span>
              <span>12.4ms</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
