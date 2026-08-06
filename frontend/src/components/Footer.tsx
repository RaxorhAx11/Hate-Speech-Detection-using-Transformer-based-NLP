import React from "react";
import { useApp } from "../context/AppContext";

export const Footer: React.FC = () => {
  const { setActiveTab } = useApp();

  return (
    <footer className="w-full bg-brand-primary text-white border-t border-hairline/10 py-12 px-6 md:px-8 mt-auto">
      <div className="mx-auto max-w-7xl">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-12 pb-8 border-b border-hairline/10">
          
          {/* Left block - Project summary */}
          <div className="col-span-1 md:col-span-6 flex flex-col justify-between">
            <div className="space-y-3">
              <span className="text-coral text-[11px] font-mono uppercase tracking-[0.28px] font-bold block">
                NLP ACADEMIC PROJECT
              </span>
              <h3 className="text-xl font-display font-light text-white leading-tight max-w-sm">
                Hate Speech Detection using Transformer NLP
              </h3>
              <p className="text-xs text-muted-slate max-w-md leading-relaxed">
                A student project designed to identify and categorize online commentaries. Powered by fine-tuned transformer model weights to detect Safe, Offensive, and Hate Speech patterns.
              </p>
            </div>
          </div>

          {/* Right Columns: Links */}
          <div className="col-span-1 md:col-span-3">
            <h4 className="text-[11px] font-mono uppercase tracking-[0.28px] text-white font-semibold mb-4">
              Core Pages
            </h4>
            <ul className="space-y-2.5 text-xs text-muted-slate font-mono">
              <li>
                <button onClick={() => setActiveTab("home")} className="hover:text-white transition-colors cursor-pointer bg-transparent border-0 p-0 text-left">
                  HOME OVERVIEW
                </button>
              </li>
              <li>
                <button onClick={() => setActiveTab("prediction")} className="hover:text-white transition-colors cursor-pointer bg-transparent border-0 p-0 text-left">
                  PREDICTION WORKBENCH
                </button>
              </li>
              <li>
                <button onClick={() => setActiveTab("history")} className="hover:text-white transition-colors cursor-pointer bg-transparent border-0 p-0 text-left">
                  PREDICTION HISTORY
                </button>
              </li>
            </ul>
          </div>

          <div className="col-span-1 md:col-span-3">
            <h4 className="text-[11px] font-mono uppercase tracking-[0.28px] text-white font-semibold mb-4">
              Documentation
            </h4>
            <ul className="space-y-2.5 text-xs text-muted-slate font-mono">
              <li>
                <button onClick={() => setActiveTab("about")} className="hover:text-white transition-colors cursor-pointer bg-transparent border-0 p-0 text-left">
                  ABOUT THE PROJECT
                </button>
              </li>
              <li>
                <button onClick={() => setActiveTab("settings")} className="hover:text-white transition-colors cursor-pointer bg-transparent border-0 p-0 text-left">
                  SYSTEM SETTINGS
                </button>
              </li>
              <li>
                <a href="https://huggingface.co/docs" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">
                  TRANSFORMER DOCS
                </a>
              </li>
            </ul>
          </div>

        </div>

        {/* Bottom microcopy */}
        <div className="flex flex-col sm:flex-row items-center justify-between pt-8 text-[11px] text-muted-slate">
          <div>
            &copy; {new Date().getFullYear()} Hate Speech Detection Project. All rights reserved.
          </div>
          <div className="mt-4 sm:mt-0 font-mono text-[10px]">
            <span>COLLEGE NLP MODEL WORKBENCH</span>
          </div>
        </div>

      </div>
    </footer>
  );
};
