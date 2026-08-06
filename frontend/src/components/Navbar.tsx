import React from "react";
import { useApp } from "../context/AppContext";

export const Navbar: React.FC = () => {
  const { activeTab, setActiveTab, isApiOnline } = useApp();

  const navItems = [
    { id: "home", label: "Home" },
    { id: "prediction", label: "Prediction" },
    { id: "history", label: "History" },
    { id: "settings", label: "Settings" },
    { id: "about", label: "About" },
  ] as const;

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border-light bg-canvas/90 backdrop-blur-md">
      {/* 36px Black Announcement Bar (from DESIGN.md component announcement-bar) */}
      <div className="flex h-9 w-full items-center justify-between bg-cohere-black px-6 text-[12px] font-medium text-white tracking-normal">
        <div className="mx-auto flex items-center gap-2">
          <span>Academic Project: Transformer-Based Content Moderation.</span>
          <button 
            onClick={() => setActiveTab("about")} 
            className="underline underline-offset-2 hover:text-coral transition-colors"
          >
            Learn more
          </button>
        </div>
      </div>

      {/* Main 3-Zone Navigation Header */}
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6 md:px-8">
        
        {/* Left Zone: Logo and Brand Anchor */}
        <div className="flex items-center gap-3">
          <button 
            onClick={() => setActiveTab("home")}
            className="flex items-center gap-2 focus-visible:ring-2 focus-visible:ring-focus-blue focus-visible:outline-none"
          >
            <div className="flex h-7 w-7 items-center justify-center rounded-sm bg-brand-primary text-white font-bold text-sm tracking-tighter">
              H
            </div>
            <span className="font-mono text-sm font-semibold tracking-[0.28px] text-brand-primary">
              HATE_SPEECH_DETECTOR
            </span>
          </button>
        </div>

        {/* Center Zone: Menu Options */}
        <nav className="hidden md:flex items-center gap-1">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`px-4 py-1.5 text-sm font-medium transition-all duration-200 rounded-sm relative focus-visible:ring-2 focus-visible:ring-focus-blue focus-visible:outline-none ${
                  isActive
                    ? "text-brand-primary font-semibold"
                    : "text-muted-slate hover:text-brand-primary"
                }`}
              >
                {item.label}
                {isActive && (
                  <span className="absolute bottom-[-16px] left-0 right-0 h-[2px] bg-brand-primary animate-fade-in" />
                )}
              </button>
            );
          })}
        </nav>

        {/* Right Zone: API Status / Call-To-Action (Primary CTA) */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1 rounded-sm border border-border-light bg-soft-stone/40 text-[11px] font-semibold font-mono">
            {isApiOnline === null ? (
              <>
                <div className="h-2 w-2 rounded-full bg-slate animate-pulse" />
                <span className="text-slate uppercase tracking-wider">Checking Server</span>
              </>
            ) : isApiOnline ? (
              <>
                <div className="h-2 w-2 rounded-full bg-deep-green" />
                <span className="text-deep-green uppercase tracking-wider">API Connected</span>
              </>
            ) : (
              <>
                <div className="h-2 w-2 rounded-full bg-error" />
                <span className="text-error uppercase tracking-wider">Server Offline</span>
              </>
            )}
          </div>

          <button
            onClick={() => setActiveTab("prediction")}
            className="hidden sm:inline-flex items-center justify-center rounded-full bg-brand-primary px-5 py-2 text-xs font-semibold text-white transition-all duration-200 hover:bg-cohere-black active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-focus-blue focus-visible:outline-none"
          >
            Start Analysis
          </button>
        </div>
      </div>

      {/* Mobile Nav Menu (simple fallback list for screen sizes smaller than md) */}
      <div className="md:hidden flex border-t border-border-light bg-canvas justify-around py-2 px-4 text-xs font-medium">
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`py-1.5 px-2.5 rounded-sm transition-colors ${
                isActive ? "text-brand-primary font-bold bg-soft-stone" : "text-muted-slate"
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </div>
    </header>
  );
};
