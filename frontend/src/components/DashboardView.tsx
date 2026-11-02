import React from 'react';
import {
  Activity,
  ShieldAlert,
  Zap,
  ArrowRight,
  Image as ImageIcon,
  Mic,
  Video,
  FileCheck2,
  Clock,
  HardDrive,
  Info,
} from 'lucide-react';
import { TabType, HistoryItem, BackendStatus } from '../types/forensics';

interface DashboardViewProps {
  onNavigate: (tab: TabType) => void;
  recentScans: HistoryItem[];
  backendStatus: BackendStatus;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  onNavigate,
  recentScans,
  backendStatus,
}) => {
  const totalScans = recentScans.length;
  const flaggedCount = recentScans.filter((s) => s.verdictDisplay === 'AI-generated' || s.verdictDisplay.includes('AI-FLAGGED')).length;
  
  // Calculate actual average latency from performed scans
  const scansWithLatency = recentScans.filter((s) => s.latencyMs !== undefined);
  const avgLatencyMs =
    scansWithLatency.length > 0
      ? Math.round(
          scansWithLatency.reduce((acc, s) => acc + (s.latencyMs || 0), 0) / scansWithLatency.length
        )
      : null;

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Backend connection banner if offline */}
      {!backendStatus.connected && (
        <div className="p-4 rounded-2xl bg-[#191B30] border border-[#A78BFA]/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-[#94A3B8]">
          <div className="flex items-center gap-2.5">
            <Info className="w-4 h-4 text-[#A78BFA] shrink-0" />
            <span>
              <strong className="text-white">Python Detector Service:</strong> The detector API is not connected on port 8000. From the backend folder, run <code className="text-[#62E8FF] bg-black/40 px-1.5 py-0.5 rounded">.\.venv\Scripts\python.exe -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000</code>.
            </span>
          </div>
          <button
            onClick={() => onNavigate('methodology')}
            className="text-[#62E8FF] hover:underline font-mono text-[11px] shrink-0"
          >
            Integration Guide &rarr;
          </button>
        </div>
      )}

      {/* Top Telemetry Metric Cards with actual calculated values */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Assets Audited */}
        <div className="bg-[#191B30]/80 border border-[#62E8FF]/30 rounded-2xl p-5 backdrop-blur-md relative overflow-hidden shadow-lg shadow-[#62E8FF]/5">
          <div className="flex items-center justify-between text-[#94A3B8] font-mono text-xs mb-2">
            <span>TOTAL ASSETS AUDITED</span>
            <div className="w-8 h-8 rounded-lg bg-[#62E8FF]/10 text-[#62E8FF] flex items-center justify-center">
              <FileCheck2 className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-[#F5F5FF] font-mono">
            {totalScans}
          </div>
          <div className="text-[11px] text-[#62E8FF] font-mono mt-1">
            {totalScans === 0 ? 'No analyses completed' : `${totalScans} evaluated in session`}
          </div>
        </div>

        {/* Flagged Media Ratio */}
        <div className="bg-[#191B30]/80 border border-[#FF75D8]/30 rounded-2xl p-5 backdrop-blur-md relative overflow-hidden shadow-lg shadow-[#FF75D8]/5">
          <div className="flex items-center justify-between text-[#94A3B8] font-mono text-xs mb-2">
            <span>FLAGGED MEDIA RATIO</span>
            <div className="w-8 h-8 rounded-lg bg-[#FF75D8]/10 text-[#FF75D8] flex items-center justify-center">
              <ShieldAlert className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-[#FF75D8] font-mono">
            {totalScans > 0 ? `${((flaggedCount / totalScans) * 100).toFixed(1)}%` : '—'}
          </div>
          <div className="text-[11px] text-[#FF75D8] font-mono mt-1">
            {totalScans > 0 ? `${flaggedCount} of ${totalScans} assets flagged` : 'Awaiting evaluations'}
          </div>
        </div>

        {/* Inference Latency */}
        <div className="bg-[#191B30]/80 border border-[#A78BFA]/30 rounded-2xl p-5 backdrop-blur-md relative overflow-hidden shadow-lg shadow-[#A78BFA]/5">
          <div className="flex items-center justify-between text-[#94A3B8] font-mono text-xs mb-2">
            <span>INFERENCE LATENCY</span>
            <div className="w-8 h-8 rounded-lg bg-[#A78BFA]/10 text-[#A78BFA] flex items-center justify-center">
              <Clock className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-[#A78BFA] font-mono">
            {avgLatencyMs !== null ? `${(avgLatencyMs / 1000).toFixed(2)}s` : '—'}
          </div>
          <div className="text-[11px] text-[#A78BFA] font-mono mt-1">
            {avgLatencyMs !== null ? 'Measured wall-clock mean' : 'Awaiting first run'}
          </div>
        </div>

        {/* Session Storage */}
        <div className="bg-[#191B30]/80 border border-[#10B981]/30 rounded-2xl p-5 backdrop-blur-md relative overflow-hidden shadow-lg shadow-[#10B981]/5">
          <div className="flex items-center justify-between text-[#94A3B8] font-mono text-xs mb-2">
            <span>SESSION STORAGE</span>
            <div className="w-8 h-8 rounded-lg bg-[#10B981]/10 text-[#10B981] flex items-center justify-center">
              <HardDrive className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-[#10B981] font-mono">
            In-Memory
          </div>
          <div className="text-[11px] text-[#10B981] font-mono mt-1">
            Client session context only
          </div>
        </div>
      </div>

      {/* Forensic Workspaces Fast Launch */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-bold text-lg text-white tracking-tight flex items-center gap-2">
            <Zap className="w-5 h-5 text-[#62E8FF]" />
            Forensic Laboratories
          </h3>
          <span className="font-mono text-xs text-[#94A3B8]">THREE DETECTION MODALITIES</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Image Lab Card */}
          <div
            onClick={() => onNavigate('image')}
            className="group relative rounded-2xl bg-[#191B30]/80 hover:bg-[#222640] border border-[#62E8FF]/30 hover:border-[#62E8FF] p-6 transition-all duration-300 cursor-pointer shadow-lg hover:shadow-[0_0_25px_rgba(98,232,255,0.25)] flex flex-col justify-between"
          >
            <div>
              <div className="w-12 h-12 rounded-xl bg-[#62E8FF]/10 border border-[#62E8FF]/30 flex items-center justify-center text-[#62E8FF] mb-4 group-hover:scale-110 transition-transform">
                <ImageIcon className="w-6 h-6" />
              </div>
              <h4 className="text-lg font-bold text-white group-hover:text-[#62E8FF] transition-colors mb-2">
                IMAGE FORENSICS
              </h4>
              <p className="text-xs text-[#94A3B8] leading-relaxed mb-4">
                ViT-Small/16 classifier evaluates 576 non-overlapping 16&times;16 spatial patches extracted from a 384&times;384 normalized image tensor.
              </p>
            </div>
            <div className="pt-3 border-t border-white/5 flex items-center justify-between font-mono text-xs text-[#62E8FF]">
              <span>LAUNCH LAB</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Audio Lab Card */}
          <div
            onClick={() => onNavigate('audio')}
            className="group relative rounded-2xl bg-[#191B30]/80 hover:bg-[#222640] border border-[#A78BFA]/30 hover:border-[#A78BFA] p-6 transition-all duration-300 cursor-pointer shadow-lg hover:shadow-[0_0_25px_rgba(167,139,250,0.25)] flex flex-col justify-between"
          >
            <div>
              <div className="w-12 h-12 rounded-xl bg-[#A78BFA]/10 border border-[#A78BFA]/30 flex items-center justify-center text-[#A78BFA] mb-4 group-hover:scale-110 transition-transform">
                <Mic className="w-6 h-6" />
              </div>
              <h4 className="text-lg font-bold text-white group-hover:text-[#A78BFA] transition-colors mb-2">
                AUDIO FORENSICS
              </h4>
              <p className="text-xs text-[#94A3B8] leading-relaxed mb-4">
                PellaV2 detector analyzes a 4.0-second central audio sample resampled via FFmpeg to 16,000 Hz 1-channel mono PCM.
              </p>
            </div>
            <div className="pt-3 border-t border-white/5 flex items-center justify-between font-mono text-xs text-[#A78BFA]">
              <span>LAUNCH LAB</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Video Lab Card */}
          <div
            onClick={() => onNavigate('video')}
            className="group relative rounded-2xl bg-[#191B30]/80 hover:bg-[#222640] border border-[#FF75D8]/30 hover:border-[#FF75D8] p-6 transition-all duration-300 cursor-pointer shadow-lg hover:shadow-[0_0_25px_rgba(255,117,216,0.25)] flex flex-col justify-between"
          >
            <div>
              <div className="w-12 h-12 rounded-xl bg-[#FF75D8]/10 border border-[#FF75D8]/30 flex items-center justify-center text-[#FF75D8] mb-4 group-hover:scale-110 transition-transform">
                <Video className="w-6 h-6" />
              </div>
              <h4 className="text-lg font-bold text-white group-hover:text-[#FF75D8] transition-colors mb-2">
                VIDEO FORENSICS
              </h4>
              <p className="text-xs text-[#94A3B8] leading-relaxed mb-4">
                Uniform temporal sampling of up to eight keyframes evaluated with ViT, coupled with demuxed audio analyzed independently.
              </p>
            </div>
            <div className="pt-3 border-t border-white/5 flex items-center justify-between font-mono text-xs text-[#FF75D8]">
              <span>LAUNCH LAB</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>
      </div>

      {/* Recent Forensic Audits Feed */}
      <div className="rounded-2xl bg-[#191B30]/80 border border-white/10 p-6 backdrop-blur-xl">
        <div className="flex items-center justify-between pb-4 border-b border-white/10 mb-4">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[#62E8FF]" />
            <h4 className="font-bold text-sm text-white tracking-tight">
              Recent Forensic Audits ({recentScans.length})
            </h4>
          </div>
          {recentScans.length > 0 && (
            <button
              onClick={() => onNavigate('history')}
              className="font-mono text-xs text-[#A78BFA] hover:text-[#FF75D8] transition-colors cursor-pointer"
            >
              View Full Audit Trail &rarr;
            </button>
          )}
        </div>

        {recentScans.length > 0 ? (
          <div className="space-y-2.5">
            {recentScans.slice(0, 5).map((scan) => {
              const isAi = scan.verdictDisplay === 'AI-generated' || scan.verdictDisplay.includes('AI-FLAGGED');
              return (
                <div
                  key={scan.id}
                  className="flex items-center justify-between p-3.5 rounded-xl bg-[#121424] border border-white/5 hover:border-white/10 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-mono font-bold ${
                        scan.mediaType === 'image'
                          ? 'bg-[#62E8FF]/10 text-[#62E8FF]'
                          : scan.mediaType === 'audio'
                          ? 'bg-[#A78BFA]/10 text-[#A78BFA]'
                          : 'bg-[#FF75D8]/10 text-[#FF75D8]'
                      }`}
                    >
                      {scan.mediaType.toUpperCase().slice(0, 3)}
                    </div>
                    <div>
                      <div className="text-xs font-bold text-white truncate max-w-xs md:max-w-md">
                        {scan.fileName}
                      </div>
                      <div className="font-mono text-[10px] text-[#64748B]">
                        {scan.timestamp} &bull; {scan.details}
                      </div>
                    </div>
                  </div>

                  <div className="text-right">
                    <span
                      className={`font-mono text-[10px] font-bold px-2.5 py-1 rounded uppercase ${
                        isAi
                          ? 'bg-[#FF75D8]/20 text-[#FF75D8] border border-[#FF75D8]/40'
                          : 'bg-[#62E8FF]/20 text-[#62E8FF] border border-[#62E8FF]/40'
                      }`}
                    >
                      {scan.verdictDisplay} ({scan.scoreDisplay})
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-10 text-xs text-[#64748B] font-mono">
            No audits recorded in this session yet. Upload a media file in any lab to begin.
          </div>
        )}
      </div>
    </div>
  );
};
