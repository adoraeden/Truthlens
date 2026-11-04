import React from 'react';
import { Sparkles, Zap, Shield, Eye, ArrowRight, Flame } from 'lucide-react';
import { TabType } from '../types/forensics';

interface HeroProps {
  onQuickStart: (tab: TabType) => void;
}

export const Hero: React.FC<HeroProps> = ({ onQuickStart }) => {
  return (
    <section className="relative rounded-3xl p-7 md:p-10 mb-8 bg-gradient-to-br from-[#191B30]/90 via-[#15172C]/95 to-[#0E101F] border border-[#62E8FF]/30 shadow-[0_12px_45px_-10px_rgba(98,232,255,0.18)] overflow-hidden backdrop-blur-xl">
      {/* Top illuminated line */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#62E8FF] via-40% to-[#FF75D8]" />

      {/* Decorative cyber grid in hero corner */}
      <div className="absolute right-0 top-0 w-80 h-80 opacity-15 pointer-events-none cyber-grid" />
      <div className="absolute -right-10 -bottom-10 w-64 h-64 rounded-full bg-[#FF75D8]/10 blur-[90px] pointer-events-none" />

      <div className="relative z-10 max-w-4xl">
        <div className="flex flex-wrap items-center gap-2 mb-3.5">
          <span className="font-mono text-xs font-bold px-3 py-1 rounded-md bg-[#62E8FF]/10 border border-[#62E8FF]/40 text-[#62E8FF] uppercase tracking-wider flex items-center gap-1.5 shadow-[0_0_15px_rgba(98,232,255,0.15)]">
            <Zap className="w-3.5 h-3.5" />
            AI Media Forensics Engine
          </span>
          <span className="font-mono text-xs font-bold px-3 py-1 rounded-md bg-[#FF75D8]/10 border border-[#FF75D8]/40 text-[#FF75D8] uppercase tracking-wider">
            Cyber Candy Identity
          </span>
          <span className="font-mono text-xs font-bold px-3 py-1 rounded-md bg-[#A78BFA]/10 border border-[#A78BFA]/40 text-[#A78BFA] uppercase tracking-wider">
            ViT + Pella V2 Stack
          </span>
        </div>

        <h1 className="text-3xl sm:text-5xl lg:text-6xl font-black tracking-tight mb-2.5 leading-tight">
          <span className="bg-gradient-to-r from-white via-[#62E8FF] via-50% to-[#FF75D8] bg-clip-text text-transparent">
            TRUTHLENS
          </span>
        </h1>

        <h2 className="text-xl sm:text-3xl font-bold text-[#A78BFA] mb-4 tracking-tight flex items-center gap-2">
          &ldquo;Reality deserves verification.&rdquo;
        </h2>

        <p className="text-sm sm:text-base text-[#94A3B8] leading-relaxed max-w-2xl mb-7">
          A prototype screening tool for images, speech recordings, and video. It reports outputs from separate image and audio detectors for human review; results may be wrong and do not establish authenticity or intent.
        </p>

        {/* Quick Launch Buttons into the three labs */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => onQuickStart('image')}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-[#62E8FF]/20 to-[#62E8FF]/5 text-[#62E8FF] border border-[#62E8FF]/50 hover:border-[#62E8FF] text-xs font-bold uppercase tracking-wider flex items-center gap-2 hover:shadow-[0_0_20px_rgba(98,232,255,0.3)] transition-all cursor-pointer"
          >
            <span>Launch Image Lab</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => onQuickStart('audio')}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-[#A78BFA]/20 to-[#A78BFA]/5 text-[#A78BFA] border border-[#A78BFA]/50 hover:border-[#A78BFA] text-xs font-bold uppercase tracking-wider flex items-center gap-2 hover:shadow-[0_0_20px_rgba(167,139,250,0.3)] transition-all cursor-pointer"
          >
            <span>Launch Audio Lab</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => onQuickStart('video')}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-[#FF75D8]/20 to-[#FF75D8]/5 text-[#FF75D8] border border-[#FF75D8]/50 hover:border-[#FF75D8] text-xs font-bold uppercase tracking-wider flex items-center gap-2 hover:shadow-[0_0_20px_rgba(255,117,216,0.3)] transition-all cursor-pointer"
          >
            <span>Launch Video Lab</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </section>
  );
};
