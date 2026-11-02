import React from 'react';
import { Cpu, Terminal, Shield, Sparkles } from 'lucide-react';

interface ScanningOverlayProps {
  currentPhase: string;
  progressPercent: number;
  mediaType: 'image' | 'audio' | 'video';
  modelLabel: string;
}

export const ScanningOverlay: React.FC<ScanningOverlayProps> = ({
  currentPhase,
  progressPercent,
  mediaType,
  modelLabel,
}) => {
  return (
    <div className="relative rounded-2xl bg-[#0F101E] border border-[#62E8FF] shadow-[0_0_40px_rgba(98,232,255,0.25)] p-8 text-center overflow-hidden my-6">
      {/* Laser scanner bar traversing across */}
      <div className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#62E8FF] to-transparent animate-laser pointer-events-none" />

      {/* Cyber radar circle */}
      <div className="relative w-24 h-24 mx-auto mb-5 flex items-center justify-center">
        <div className="absolute inset-0 rounded-full border border-[#62E8FF]/30" />
        <div className="absolute inset-2 rounded-full border border-dashed border-[#A78BFA]/40 animate-radar" />
        <div className="absolute inset-5 rounded-full border border-[#FF75D8]/40" />
        <div className="w-4 h-4 rounded-full bg-[#62E8FF] shadow-[0_0_15px_#62E8FF] animate-pulse" />
      </div>

      <div className="font-mono text-sm md:text-base font-bold text-[#62E8FF] tracking-widest uppercase mb-1">
        {currentPhase || 'EXECUTING FORENSIC DEEP SCAN...'}
      </div>

      <div className="text-xs text-[#94A3B8] font-mono flex items-center justify-center gap-2 mb-4">
        <Cpu className="w-3.5 h-3.5 text-[#A78BFA]" />
        <span>BACKBONE: {modelLabel}</span>
        <span>&bull;</span>
        <span className="text-[#10B981]">ISOLATED CPU THREADS [8]</span>
      </div>

      {/* Progress bar */}
      <div className="max-w-md mx-auto">
        <div className="flex justify-between font-mono text-[10px] text-[#64748B] mb-1">
          <span>PIPELINE DISPATCH</span>
          <span className="text-[#62E8FF]">{progressPercent}%</span>
        </div>
        <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
          <div
            style={{ width: `${progressPercent}%` }}
            className="h-full bg-gradient-to-r from-[#62E8FF] via-[#A78BFA] to-[#FF75D8] transition-all duration-300"
          />
        </div>
      </div>
    </div>
  );
};
