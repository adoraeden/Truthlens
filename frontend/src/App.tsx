import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Hero } from './components/Hero';
import { DashboardView } from './components/DashboardView';
import { ImageLab } from './components/ImageLab';
import { AudioLab } from './components/AudioLab';
import { VideoLab } from './components/VideoLab';
import { HistoryView } from './components/HistoryView';
import { MethodologyView } from './components/MethodologyView';
import { TabType, HistoryItem, BackendStatus } from './types/forensics';
import { checkBackendHealth } from './services/api';
import { Layers, Shield, Cpu, Info } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  
  // Actual session audit history (starts completely empty, no fake records)
  const [history, setHistory] = useState<HistoryItem[]>([]);

  const [backendStatus, setBackendStatus] = useState<BackendStatus>({
    connected: false,
    endpoint: '/api',
    message: 'Checking Python detector service connection...',
  });

  // Probe backend health periodically
  useEffect(() => {
    let isMounted = true;
    const probe = async () => {
      const status = await checkBackendHealth();
      if (isMounted) {
        setBackendStatus(status);
      }
    };
    probe();
    const interval = setInterval(probe, 8000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleScanCompleted = (item: HistoryItem) => {
    setHistory((prev) => [item, ...prev]);
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-screen bg-[#10111F] text-[#F5F5FF] flex flex-col md:flex-row relative selection:bg-[#FF75D8]/30 selection:text-white">
      {/* Ambient Glowing Atmospheric Orbs */}
      <div className="fixed top-[-10%] left-[-5%] w-[45vw] h-[45vw] rounded-full bg-[#62E8FF]/6 blur-[140px] pointer-events-none" />
      <div className="fixed top-[25%] right-[-10%] w-[50vw] h-[50vw] rounded-full bg-[#FF75D8]/6 blur-[160px] pointer-events-none" />
      <div className="fixed bottom-[-10%] left-[25%] w-[40vw] h-[40vw] rounded-full bg-[#A78BFA]/5 blur-[140px] pointer-events-none" />

      {/* SIDEBAR NAVIGATION */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={(tab) => setActiveTab(tab)}
        backendStatus={backendStatus}
        historyCount={history.length}
      />

      {/* MAIN VIEWPORT */}
      <main className="flex-1 p-4 md:p-8 max-w-6xl mx-auto w-full z-10 overflow-y-auto">
        {/* HERO SECTION (Shown on Dashboard) */}
        {activeTab === 'dashboard' && <Hero onQuickStart={(tab) => setActiveTab(tab)} />}

        {/* WORKSPACES */}
        {activeTab === 'dashboard' && (
          <DashboardView
            onNavigate={(tab) => setActiveTab(tab)}
            recentScans={history}
            backendStatus={backendStatus}
          />
        )}

        {activeTab === 'image' && (
          <ImageLab
            onScanCompleted={handleScanCompleted}
            backendStatus={backendStatus}
          />
        )}

        {activeTab === 'audio' && (
          <AudioLab
            onScanCompleted={handleScanCompleted}
            backendStatus={backendStatus}
          />
        )}

        {activeTab === 'video' && (
          <VideoLab
            onScanCompleted={handleScanCompleted}
            backendStatus={backendStatus}
          />
        )}

        {activeTab === 'history' && (
          <HistoryView history={history} onClearHistory={handleClearHistory} />
        )}

        {activeTab === 'methodology' && (
          <MethodologyView backendStatus={backendStatus} />
        )}

        {/* TECHNICALLY ACCURATE DOSSIER FOOTNOTE */}
        <section className="mt-10 rounded-2xl bg-[#14162B]/80 border border-white/5 p-6 backdrop-blur-md">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/5">
            <div className="font-bold text-sm text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#62E8FF]" />
              TRUTHLENS OPERATIONAL SUMMARY &amp; INTEGRITY PRINCIPLES
            </div>
            <button
              onClick={() => setActiveTab('methodology')}
              className="font-mono text-[10px] text-[#62E8FF] hover:underline cursor-pointer"
            >
              Full Technical Methodology &rarr;
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-[#94A3B8] leading-relaxed">
            <div>
              <div className="font-bold text-[#62E8FF] mb-1.5 uppercase font-mono">
                Image Modality (ViT-Small/16)
              </div>
              Fine-tuned ViT-Small/16 classifier. Resamples inputs to 384&times;384 and evaluates 576 non-overlapping 16&times;16 spatial patches against a logit cutoff threshold of 1.359375.
            </div>
            <div>
              <div className="font-bold text-[#A78BFA] mb-1.5 uppercase font-mono">
                Audio Modality (PellaV2)
              </div>
              PellaV2 detector with Wav2Vec-BERT backbone. FFmpeg converts audio to 16,000 Hz mono PCM and evaluates a four-second central crop with a 0.50 decision threshold.
            </div>
            <div>
              <div className="font-bold text-[#FF75D8] mb-1.5 uppercase font-mono">
                Core Constraints &amp; Boundaries
              </div>
              Model scores are uncalibrated indices, not probabilities. Real media may be falsely flagged by compression; re-recorded speech and unsampled video frames evade detection.
            </div>
          </div>
        </section>

        {/* FOOTER */}
        <footer className="mt-8 pt-6 border-t border-white/5 text-center font-mono text-[11px] text-[#64748B]">
          TRUTHLENS &bull; MULTIMODAL MEDIA FORENSICS &bull; CYBER CANDY INTERFACE
        </footer>
      </main>
    </div>
  );
}
