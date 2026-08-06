import React from "react";
import { Navbar } from "./Navbar";
import { Footer } from "./Footer";

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col bg-canvas text-ink">
      {/* Dynamic Navbar */}
      <Navbar />

      {/* Main Content Area with Editorial Restraint */}
      <main className="flex-1 w-full mx-auto max-w-7xl px-6 md:px-8 py-10 md:py-16">
        {children}
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
};
