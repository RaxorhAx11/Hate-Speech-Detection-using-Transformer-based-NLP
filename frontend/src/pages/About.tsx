import React from "react";
import { BookOpen, HelpCircle, Layers, Cpu, BarChart3 } from "lucide-react";
import { useApp } from "../context/AppContext";

export const About: React.FC = () => {
  const { metrics } = useApp();

  // Fallback real metrics from training evaluation run
  const fallbackMetrics = {
    accuracy: 0.8433,
    precision_macro: 0.8457,
    recall_macro: 0.8445,
    f1_macro: 0.8434,
    roc_auc_macro: 0.9467,
    class_metrics: {
      Safe: { precision: 0.90, recall: 0.90, f1_score: 0.90, support: 484 },
      Offensive: { precision: 0.83, recall: 0.75, f1_score: 0.79, support: 511 },
      "Hate Speech": { precision: 0.80, recall: 0.89, f1_score: 0.84, support: 505 }
    }
  };

  const activeMetrics = metrics?.available ? metrics : fallbackMetrics;
  return (
    <div className="max-w-4xl mx-auto space-y-12">
      {/* Page Header */}
      <div className="space-y-3">
        <span className="text-[11px] font-mono uppercase tracking-[0.28px] text-slate block">
          TECHNICAL RESEARCH DOCUMENTATION
        </span>
        <h1 className="text-4xl font-display font-light text-brand-primary tracking-[-0.48px]">
          About the Project
        </h1>
        <p className="text-sm text-body-muted leading-relaxed">
          Learn about the dataset compilation pipeline, transformer modeling configs, and model evaluation metrics.
        </p>
      </div>

      {/* Main content grid */}
      <div className="space-y-12">
        
        {/* Project Section */}
        <section className="space-y-4 border-t border-hairline pt-8">
          <div className="flex items-center gap-2">
            <BookOpen className="h-4.5 w-4.5 text-brand-primary" />
            <h2 className="text-lg font-mono uppercase font-bold tracking-wide text-brand-primary">
              1. Project Overview
            </h2>
          </div>
          <p className="text-sm text-body-muted leading-relaxed">
            Mitigating toxic interactions in online public spaces is critical for constructive community spaces. This project implements a production-grade content moderation pipeline designed to scan text inputs for offensive content and targeted attacks. By leveraging transformer neural networks, the system maps statements into three categories: Safe (clean comment), Offensive (general toxic remarks), and Hate Speech (targeted attacks against protected classes, insults, or threats of violence).
          </p>
        </section>

        {/* Dataset compilation section */}
        <section className="space-y-4 border-t border-hairline pt-8">
          <div className="flex items-center gap-2">
            <HelpCircle className="h-4.5 w-4.5 text-brand-primary" />
            <h2 className="text-lg font-mono uppercase font-bold tracking-wide text-brand-primary">
              2. Unified Dataset Mapping
            </h2>
          </div>
          <p className="text-sm text-body-muted leading-relaxed">
            Training a robust classifier requires high-quality, diverse data. We unify, clean, and stratify annotator comments across **five** public toxicity source datasets to construct our dataset split:
          </p>

          {/* Table mapping source datasets (GFM layout) */}
          <div className="overflow-x-auto border border-border-light rounded-sm bg-canvas">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-soft-stone/30 border-b border-border-light">
                  <th className="p-3 font-mono uppercase font-bold tracking-wider text-slate">Source Dataset</th>
                  <th className="p-3 font-mono uppercase font-bold tracking-wider text-slate">Original Class Schema</th>
                  <th className="p-3 font-mono uppercase font-bold tracking-wider text-slate">Unified Target Mapping</th>
                  <th className="p-3 font-mono uppercase font-bold tracking-wider text-slate">Description</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hairline">
                <tr>
                  <td className="p-3 font-semibold text-brand-primary">Davidson et al.</td>
                  <td className="p-3 font-mono">0 (hate) / 1 (offensive) / 2 (neither)</td>
                  <td className="p-3 font-semibold text-coral">Safe / Offensive / Hate Speech</td>
                  <td className="p-3 text-body-muted">Standard 3-class alignment.</td>
                </tr>
                <tr>
                  <td className="p-3 font-semibold text-brand-primary">OLID (Subtask A)</td>
                  <td className="p-3 font-mono">OFF (offensive) / NOT (not offensive)</td>
                  <td className="p-3 font-semibold text-coral">Safe / Offensive</td>
                  <td className="p-3 text-body-muted">General toxicity check, no explicit hate class.</td>
                </tr>
                <tr>
                  <td className="p-3 font-semibold text-brand-primary">HateXplain</td>
                  <td className="p-3 font-mono">0 (hatespeech) / 1 (normal) / 2 (offensive)</td>
                  <td className="p-3 font-semibold text-coral">Safe / Offensive / Hate Speech</td>
                  <td className="p-3 text-body-muted">Consolidated via majority annotator vote.</td>
                </tr>
                <tr>
                  <td className="p-3 font-semibold text-brand-primary">Jigsaw Toxicity</td>
                  <td className="p-3 font-mono">multi-label toxicity indicators</td>
                  <td className="p-3 font-semibold text-coral">Safe / Offensive / Hate Speech</td>
                  <td className="p-3 text-body-muted">Mapped using threshold flags for identity attacks/threats.</td>
                </tr>
                <tr>
                  <td className="p-3 font-semibold text-brand-primary">Civil Comments</td>
                  <td className="p-3 font-mono">toxicity rate values (0.0 to 1.0)</td>
                  <td className="p-3 font-semibold text-coral">Safe / Offensive / Hate Speech</td>
                  <td className="p-3 text-body-muted">Mapped using severity scoring (threshold: 0.5).</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Model section */}
        <section className="space-y-4 border-t border-hairline pt-8">
          <div className="flex items-center gap-2">
            <Cpu className="h-4.5 w-4.5 text-brand-primary" />
            <h2 className="text-lg font-mono uppercase font-bold tracking-wide text-brand-primary">
              3. Model Fine-Tuning & Inference
            </h2>
          </div>
          <p className="text-sm text-body-muted leading-relaxed">
            The model defaults to <strong>DistilBERT-base-uncased</strong>—a distilled version of BERT that retains over 97% of BERT's language understanding capability while being 40% smaller and 60% faster during forward passes. Fine-tuning uses:
          </p>
          <ul className="list-disc list-inside space-y-2 text-xs text-body-muted pl-2">
            <li><strong>Class Weight Adjustments:</strong> Custom cross-entropy loss weights are applied to correct target label imbalances across datasets.</li>
            <li><strong>Early Stopping:</strong> Monitoring test macro-F1 to prevent overfitting during training epochs.</li>
            <li><strong>Deterministic Settings:</strong> Random seeds are pinned across PyTorch and Hugging Face pipelines to ensure predictions stay replicable.</li>
          </ul>
        </section>

        {/* Technical stack section */}
        <section className="space-y-4 border-t border-hairline pt-8">
          <div className="flex items-center gap-2">
            <Layers className="h-4.5 w-4.5 text-brand-primary" />
            <h2 className="text-lg font-mono uppercase font-bold tracking-wide text-brand-primary">
              4. Technology Suite
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
            <div className="bg-soft-stone p-5 rounded-sm space-y-2.5">
              <span className="font-semibold text-brand-primary block font-mono">BACKEND API ENGINE</span>
              <ul className="space-y-1.5 text-body-muted">
                <li>• PyTorch Deep Learning framework</li>
                <li>• Hugging Face Transformers tokenizers</li>
                <li>• FastAPI asynchronous endpoints</li>
                <li>• Python-SpeechRecognition (Google Voice Fallbacks)</li>
                <li>• PyTTSx3 Speech Synthesis framework</li>
              </ul>
            </div>
            <div className="bg-soft-stone p-5 rounded-sm space-y-2.5">
              <span className="font-semibold text-brand-primary block font-mono">FRONTEND INTERACTION LAYER</span>
              <ul className="space-y-1.5 text-body-muted">
                <li>• React.js (Component-driven architecture)</li>
                <li>• Vite compilation bundler</li>
                <li>• TypeScript static typings</li>
                <li>• Tailwind CSS v4 styling structure</li>
                <li>• Browser Web Speech Synthesis & Recognition APIs</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Model Evaluation Metrics section */}
        <section className="space-y-4 border-t border-hairline pt-8">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4.5 w-4.5 text-brand-primary" />
            <h2 className="text-lg font-mono uppercase font-bold tracking-wide text-brand-primary">
              5. Model Evaluation Metrics
            </h2>
          </div>
          <p className="text-sm text-body-muted leading-relaxed">
            Below are the performance metrics generated during the model testing phase on a stratified hold-out test set containing 1,500 samples.
          </p>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-soft-stone/30 border border-border-light p-4 rounded-sm text-center">
              <span className="block text-[10px] font-mono text-slate uppercase">Accuracy</span>
              <span className="text-xl font-semibold text-brand-primary">{(activeMetrics.accuracy * 100).toFixed(2)}%</span>
            </div>
            <div className="bg-soft-stone/30 border border-border-light p-4 rounded-sm text-center">
              <span className="block text-[10px] font-mono text-slate uppercase">Precision (Macro)</span>
              <span className="text-xl font-semibold text-brand-primary">{(activeMetrics.precision_macro * 100).toFixed(2)}%</span>
            </div>
            <div className="bg-soft-stone/30 border border-border-light p-4 rounded-sm text-center">
              <span className="block text-[10px] font-mono text-slate uppercase">Recall (Macro)</span>
              <span className="text-xl font-semibold text-brand-primary">{(activeMetrics.recall_macro * 100).toFixed(2)}%</span>
            </div>
            <div className="bg-soft-stone/30 border border-border-light p-4 rounded-sm text-center">
              <span className="block text-[10px] font-mono text-slate uppercase">F1-Score (Macro)</span>
              <span className="text-xl font-semibold text-brand-primary">{(activeMetrics.f1_macro * 100).toFixed(2)}%</span>
            </div>
            <div className="bg-soft-stone/30 border border-border-light p-4 rounded-sm text-center col-span-2 md:col-span-1">
              <span className="block text-[10px] font-mono text-slate uppercase">ROC AUC (Macro)</span>
              <span className="text-xl font-semibold text-brand-primary">{(activeMetrics.roc_auc_macro * 100).toFixed(2)}%</span>
            </div>
          </div>

          <div className="overflow-x-auto border border-border-light rounded-sm bg-canvas mt-4">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-soft-stone/30 border-b border-border-light">
                  <th className="p-3 font-mono uppercase font-bold tracking-wider text-slate">Class</th>
                  <th className="p-3 font-mono uppercase font-bold tracking-wider text-slate">Precision</th>
                  <th className="p-3 font-mono uppercase font-bold tracking-wider text-slate">Recall</th>
                  <th className="p-3 font-mono uppercase font-bold tracking-wider text-slate">F1-Score</th>
                  <th className="p-3 font-mono uppercase font-bold tracking-wider text-slate">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hairline">
                {(["Safe", "Offensive", "Hate Speech"] as const).map((label) => {
                  const m = activeMetrics.class_metrics[label];
                  return (
                    <tr key={label}>
                      <td className="p-3 font-semibold text-brand-primary">{label}</td>
                      <td className="p-3 font-mono">{m ? m.precision.toFixed(2) : "0.00"}</td>
                      <td className="p-3 font-mono">{m ? m.recall.toFixed(2) : "0.00"}</td>
                      <td className="p-3 font-mono">{m ? m.f1_score.toFixed(2) : "0.00"}</td>
                      <td className="p-3 font-mono">{m ? m.support : "0"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
};
