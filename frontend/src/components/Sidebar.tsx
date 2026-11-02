import React from 'react';
import {
  LayoutDashboard,
  Image as ImageIcon,
  Mic,
  Video,
  History,
  Layers,
  Zap,
  Shield,
} from 'lucide-react';
import { TabType, BackendStatus } from '../types/forensics';

interface SidebarProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  backendStatus: BackendStatus;
  historyCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  backendStatus,
  historyCount,
}) => {
  const navItems = [
    { id: 'dashboard' as TabType, label: 'Dashboard', icon: LayoutDashboard, badge: null },
    { id: 'image' as TabType, label: 'Image Forensics', icon: ImageIcon, badge: 'ViT-16' },
    { id: 'audio' as TabType, label: 'Audio Forensics', icon: Mic, badge: 'PellaV2' },
    { id: 'video' as TabType, label: 'Video Forensics', icon: Video, badge: 'Dual' },
    { id: 'history' as TabType, label: 'Forensic Audit Trail', icon: History, badge: historyCount > 0 ? `${historyCount}` : null },
    { id: 'methodology' as TabType, label: 'System & Methodology', icon: Layers, badge: null },
  ];

  return (
    <aside className="w-full md:w-72 bg-[#0B0C16] border-r border-[#A78BFA]/18 p-5 flex flex-col justify-between shrink-0 z-20 backdrop-blur-xl">
      <div>
        {/* Brand / Logo */}
        <div
          onClick={() => onTabChange('dashboard')}
          className="flex items-center gap-3.5 mb-7 pb-5 border-b border-white/5 cursor-pointer group"
        >
          <div className="w-11 h-11 rounded-xl bg-gradient-to-tr from-[#62E8FF] via-[#A78BFA] to-[#FF75D8] p-[1.5px] flex items-center justify-center shadow-lg shadow-[#62E8FF]/20 group-hover:scale-105 transition-transform">
            <div className="w-full h-full bg-[#10111F] rounded-[10px] flex items-center justify-center">
              <Zap className="w-6 h-6 text-[#62E8FF] filter drop-shadow-[0_0_8px_#62E8FF]" />
            </div>
          </div>
          <div>
            <div className="font-extrabold tracking-tight text-xl bg-gradient-to-r from-white via-[#62E8FF] to-[#FF75D8] bg-clip-text text-transparent">
              TRUTHLENS
            </div>
            <div className="font-mono text-[10px] text-[#A78BFA] tracking-widest uppercase font-semibold">
              CYBER CANDY FORENSICS
            </div>
          </div>
        </div>

        {/* Live System Status HUD */}
        <div className="bg-[#191B30]/80 border border-[#62E8FF]/25 rounded-2xl p-4 mb-6 shadow-md shadow-[#62E8FF]/5">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="relative flex h-2.5 w-2.5">
                <span
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                    backendStatus.connected ? 'bg-[#10B981]' : 'bg-[#A78BFA]'
                  }`}
                />
                <span
                  className={`relative inline-flex rounded-full h-2.5 w-2.5 ${
                    backendStatus.connected ? 'bg-[#10B981]' : 'bg-[#A78BFA]'
                  }`}
                />
              </span>
              <span
                className={`font-mono text-xs font-bold tracking-wider ${
                  backendStatus.connected ? 'text-[#10B981]' : 'text-[#A78BFA]'
                }`}
              >
                {backendStatus.connected ? 'BACKEND LINKED' : 'STANDBY MODE'}
              </span>
            </div>
            <span className="font-mono text-[10px] text-[#94A3B8] bg-white/5 px-2 py-0.5 rounded border border-white/10">
              V2.4.0
            </span>
          </div>

          <div className="space-y-1.5 font-mono text-[11px] text-[#94A3B8] pt-2 border-t border-white/5">
            <div className="flex justify-between items-center">
              <span>PYTHON DETECTOR:</span>
              <span
                className={`font-semibold ${
                  backendStatus.connected ? 'text-[#10B981]' : 'text-[#A78BFA]'
                }`}
              >
                {backendStatus.connected ? 'ONLINE (:8000)' : 'OFFLINE'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span>HOST THREADS:</span>
              <span className="text-[#F5F5FF] font-semibold">CPU [8 THREADS]</span>
            </div>
            <div className="flex justify-between items-center">
              <span>SESSION RETENTION:</span>
              <span className="text-[#62E8FF]">IN-MEMORY</span>
            </div>
          </div>
        </div>

        {/* Navigation Menu */}
        <div className="space-y-1.5">
          <div className="font-mono text-[10px] text-[#64748B] uppercase tracking-wider px-2.5 mb-2 font-bold">
            Workspaces &amp; Telemetry
          </div>

          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-3 rounded-xl font-medium text-sm transition-all cursor-pointer ${
                  isActive
                    ? 'bg-gradient-to-r from-[#62E8FF]/20 via-[#A78BFA]/10 to-transparent text-white border border-[#62E8FF]/50 shadow-[0_0_20px_rgba(98,232,255,0.18)] font-semibold'
                    : 'text-[#94A3B8] hover:text-white hover:bg-white/5 border border-transparent'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon
                    className={`w-4 h-4 ${
                      isActive ? 'text-[#62E8FF] drop-shadow-[0_0_6px_#62E8FF]' : 'text-[#94A3B8]'
                    }`}
                  />
                  <span>{item.label}</span>
                </div>

                {item.badge && (
                  <span
                    className={`font-mono text-[10px] px-2 py-0.5 rounded-full border ${
                      isActive
                        ? 'bg-[#62E8FF]/20 border-[#62E8FF]/40 text-[#62E8FF]'
                        : 'bg-white/5 border-white/10 text-[#64748B]'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Footer Info */}
      <div className="pt-5 mt-6 border-t border-white/5 font-mono text-[11px] text-[#64748B] space-y-1.5">
        <div className="flex items-center gap-1.5 text-[#94A3B8]">
          <Shield className="w-3.5 h-3.5 text-[#A78BFA]" />
          <span>Multimodal Media Screening</span>
        </div>
        <div className="text-[10px] leading-relaxed text-[#64748B]">
          Automated screening indicator. Not a calibrated probability.
        </div>
      </div>
    </aside>
  );
};
