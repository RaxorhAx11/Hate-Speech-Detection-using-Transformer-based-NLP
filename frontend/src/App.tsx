import React from "react";
import { AppProvider, useApp } from "./context/AppContext";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { Prediction } from "./pages/Prediction";
import { History } from "./pages/History";
import { Settings } from "./pages/Settings";
import { About } from "./pages/About";

// Inner router that accesses context
const AppContent: React.FC = () => {
  const { activeTab } = useApp();

  const renderContent = () => {
    switch (activeTab) {
      case "home":
        return <Home />;
      case "prediction":
        return <Prediction />;
      case "history":
        return <History />;
      case "settings":
        return <Settings />;
      case "about":
        return <About />;
      default:
        return <Home />;
    }
  };

  return <Layout>{renderContent()}</Layout>;
};

// Global App wrapper providing state
export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}
