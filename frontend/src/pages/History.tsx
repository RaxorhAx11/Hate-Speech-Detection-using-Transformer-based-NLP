import React, { useState } from "react";
import { useApp } from "../context/AppContext";
import { Search, Trash2, Download, AlertCircle, FileSpreadsheet, Calendar } from "lucide-react";

export const History: React.FC = () => {
  const { history, deleteFromHistory, clearHistory } = useApp();
  const [searchQuery, setSearchQuery] = useState("");
  const [activeFilter, setActiveFilter] = useState<"all" | "Safe" | "Offensive" | "Hate Speech">("all");

  const filteredHistory = history.filter((item) => {
    const matchesSearch = item.text.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = activeFilter === "all" || item.prediction === activeFilter;
    return matchesSearch && matchesFilter;
  });

  const getPillColor = (label: string) => {
    switch (label) {
      case "Safe":
        return "text-deep-green bg-pale-green border-deep-green/20";
      case "Offensive":
        return "text-coral bg-coral-soft/20 border-coral-soft/50";
      case "Hate Speech":
        return "text-error bg-error/15 border-error/30";
      default:
        return "text-slate bg-soft-stone border-hairline";
    }
  };

  const exportToJson = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `prediction_history_${new Date().toISOString().slice(0,10)}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const exportToCsv = () => {
    // Construct CSV Header & Rows
    const headers = ["Timestamp", "Comment text", "Prediction", "Confidence", "Safe Prob", "Offensive Prob", "Hate Speech Prob", "Processing Time MS"];
    const rows = history.map((item) => [
      item.timestamp,
      `"${item.text.replace(/"/g, '""')}"`,
      item.prediction,
      item.confidence,
      item.probabilities.Safe,
      item.probabilities.Offensive,
      item.probabilities["Hate Speech"],
      item.processingTimeMs
    ]);

    const csvContent = "data:text/csv;charset=utf-8," 
      + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
      
    const encodedUri = encodeURI(csvContent);
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", encodedUri);
    downloadAnchor.setAttribute("download", `prediction_history_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-2">
          <span className="text-[11px] font-mono uppercase tracking-[0.28px] text-slate block">
            HISTORICAL AUDIT LOGS
          </span>
          <h1 className="text-4xl font-display font-light text-brand-primary tracking-[-0.48px]">
            Prediction Registry
          </h1>
          <p className="text-sm text-body-muted leading-relaxed">
            Review past toxic analysis scans saved locally. Filter items, clear histories, or export structured datasets for model benchmarking.
          </p>
        </div>

        {/* Global actions: Clear, Export */}
        {history.length > 0 && (
          <div className="flex items-center gap-2 text-xs">
            <button
              onClick={exportToCsv}
              className="inline-flex items-center gap-1.5 px-4 py-2 border border-border-light hover:bg-soft-stone rounded-sm font-medium transition-colors"
            >
              <FileSpreadsheet className="h-3.5 w-3.5" />
              <span>Export CSV</span>
            </button>
            <button
              onClick={exportToJson}
              className="inline-flex items-center gap-1.5 px-4 py-2 border border-border-light hover:bg-soft-stone rounded-sm font-medium transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Export JSON</span>
            </button>
            <button
              onClick={() => {
                if (window.confirm("Are you sure you want to clear all history records?")) {
                  clearHistory();
                }
              }}
              className="inline-flex items-center gap-1.5 px-4 py-2 border border-error/30 hover:bg-error/15 text-error rounded-sm font-medium transition-colors"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Clear Registry</span>
            </button>
          </div>
        )}
      </div>

      {/* Filter and Search Bar (from DESIGN.md component blog-filter-chip style) */}
      <div className="flex flex-col md:flex-row md:items-center gap-4 justify-between border-b border-hairline/60 pb-6">
        
        {/* Outlined pills for category filters */}
        <div className="flex flex-wrap items-center gap-2">
          {(["all", "Safe", "Offensive", "Hate Speech"] as const).map((filter) => {
            const isActive = activeFilter === filter;
            return (
              <button
                key={filter}
                onClick={() => setActiveFilter(filter)}
                className={`px-4 py-1.5 rounded-full border text-xs font-mono tracking-wide uppercase transition-all ${
                  isActive
                    ? "bg-brand-primary text-white border-brand-primary font-semibold"
                    : "bg-transparent text-slate border-hairline hover:border-slate hover:text-brand-primary"
                }`}
              >
                {filter === "all" ? "SHOW ALL" : filter}
              </button>
            );
          })}
        </div>

        {/* Search Input Box */}
        <div className="relative w-full md:w-80">
          <Search className="absolute left-3.5 top-2.5 h-4 w-4 text-slate" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search registry text..."
            className="w-full text-xs pl-10 pr-4 py-2.5 border border-border-light rounded-sm outline-none focus:border-form-focus transition-colors"
          />
        </div>
      </div>

      {/* History List Table (DESIGN.md research-table style) */}
      {filteredHistory.length > 0 ? (
        <div className="border border-border-light rounded-sm overflow-hidden bg-canvas">
          <div className="divide-y divide-hairline">
            {filteredHistory.map((item) => (
              <div
                key={item.id}
                className="p-5 md:p-6 flex flex-col md:flex-row md:items-center justify-between gap-4 hover:bg-soft-stone/10 transition-colors"
              >
                
                {/* Text commentary and timestamp */}
                <div className="flex-1 space-y-2">
                  <p className="text-sm font-medium text-brand-primary leading-relaxed pr-6">
                    {item.text}
                  </p>
                  <div className="flex items-center gap-4 text-[10px] text-slate font-mono">
                    <span className="flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      {new Date(item.timestamp).toLocaleString()}
                    </span>
                    <span>LATENCY: {item.processingTimeMs.toFixed(1)}ms</span>
                  </div>
                </div>

                {/* Prediction tag and Deletion actions */}
                <div className="flex items-center justify-between md:justify-end gap-6 shrink-0 border-t md:border-t-0 pt-3 md:pt-0">
                  <div className="flex items-center gap-2">
                    <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-medium border ${getPillColor(item.prediction)}`}>
                      {item.prediction}
                    </span>
                    <span className="text-[11px] font-mono text-slate">
                      {item.confidence.toFixed(1)}%
                    </span>
                  </div>

                  <button
                    onClick={() => deleteFromHistory(item.id)}
                    className="p-2 text-slate hover:text-error hover:bg-error/10 rounded-full transition-colors focus-visible:ring-1 focus-visible:ring-focus-blue"
                    title="Delete record"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>

              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="text-center py-20 border border-dashed border-hairline/80 rounded-sm bg-soft-stone/10">
          <AlertCircle className="h-8 w-8 text-slate mx-auto mb-4" />
          <h3 className="text-base font-medium text-brand-primary mb-1">
            No history files matches your request
          </h3>
          <p className="text-xs text-muted-slate max-w-sm mx-auto">
            {history.length === 0
              ? "Predictions will be registered automatically once commentary is submitted."
              : "Try altering search terms or selecting another category filter above."}
          </p>
        </div>
      )}
    </div>
  );
};
