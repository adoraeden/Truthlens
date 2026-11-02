import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, Info } from 'lucide-react';
import { Verdict } from '../types/forensics';

interface VerdictBannerProps {
  verdict: Verdict;
  score: number;
  subtitle?: string;
  sourceLabel?: string;
  latencyMs?: number;
}

export const VerdictBanner: React.FC<VerdictBannerProps> = ({
  verdict,
  score,
  subtitle,
  sourceLabel = 'Python Neural Detector',
  latencyMs,
}) => {
  const isAI = verdict === 'AI-generated';
  const scorePct = Math.round(score * 10000) / 100;

  return (
    <div
      className={`relative rounded-2xl p-6 md:p-8 border transition-all overflow-hidden ${
        isAI
          ? 'bg-gradient-to-br from-[#FF3366]/20 via-[#191B30] to-[#10111F] border-[#FF3366]/70 shadow-[0_10px_45px_-10px_rgba(255,51,102,0.45)]'
          : 'bg-gradient-to-br from-[#10B981]/20 via-[#191B30] to-[#10111F] border-[#10B981]/70 shadow-[0_10px_45px_-10px_rgba(16,185,129,0.35)]'
      }`}
    >
      {/* Top glowing edge line */}
      <div
        className={`absolute top-0 left-0 right-0 h-[2px] ${
          isAI
            ? 'bg-gradient-to-r from-transparent via-[#FF75D8] to-transparent'
            : 'bg-gradient-to-r from-transparent via-[#62E8FF] to-transparent'
        }`}
      />

      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 mb-6">
        <div>
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span
              className={`font-mono text-xs font-bold px-2.5 py-1 rounded-md uppercase tracking-wider border ${
                isAI
                  ? 'bg-[#FF75D8]/10 text-[#FF75D8] border-[#FF75D8]/40'
                  : 'bg-[#62E8FF]/10 text-[#62E8FF] border-[#62E8FF]/40'
              }`}
            >
              FORENSIC VERDICT
            </span>
            <span className="font-mono text-[10px] text-[#A78BFA] bg-[#A78BFA]/10 border border-[#A78BFA]/30 px-2 py-0.5 rounded">
              {sourceLabel}
            </span>
            {latencyMs !== undefined && (
              <span className="font-mono text-[10px] text-[#62E8FF] bg-[#62E8FF]/10 border border-[#62E8FF]/30 px-2 py-0.5 rounded">
                LATENCY: {(latencyMs / 1000).toFixed(2)}s
              </span>
            )}
          </div>

          <div className="flex items-center gap-3">
            {isAI ? (
              <>
                <div className="w-10 h-10 rounded-xl bg-[#FF3366]/20 border border-[#FF3366]/50 flex items-center justify-center shrink-0 shadow-[0_0_20px_rgba(255,51,102,0.4)]">
                  <ShieldAlert className="w-6 h-6 text-[#FF75D8]" />
                </div>
                <h3 className="text-2xl md:text-3xl font-extrabold text-[#FF75D8] tracking-tight">
                  AI-GENERATED INDICATORS DETECTED
                </h3>
              </>
            ) : (
              <>
                <div className="w-10 h-10 rounded-xl bg-[#10B981]/20 border border-[#10B981]/50 flex items-center justify-center shrink-0 shadow-[0_0_20px_rgba(16,185,129,0.4)]">
                  <ShieldCheck className="w-6 h-6 text-[#62E8FF]" />
                </div>
                <h3 className="text-2xl md:text-3xl font-extrabold text-[#62E8FF] tracking-tight">
                  LIKELY REAL MEDIA
                </h3>
              </>
            )}
          </div>

          <p className="text-xs md:text-sm text-[#94A3B8] mt-2 max-w-2xl leading-relaxed">
            {subtitle ||
              (isAI
                ? 'Model output crossed the detection threshold based on neural latent representations.'
                : 'Model output remained below the synthetic classification threshold.')}
          </p>
        </div>

        {/* Score Display */}
        <div className="lg:text-right lg:border-l lg:border-white/10 lg:pl-8 shrink-0">
          <div className="font-mono text-xs text-[#64748B] uppercase tracking-wider mb-0.5">
            UNCALIBRATED MODEL SCORE
          </div>
          <div
            className={`font-mono text-4xl md:text-5xl font-extrabold tracking-tight ${
              isAI ? 'text-[#FF75D8]' : 'text-[#62E8FF]'
            }`}
          >
            {scorePct.toFixed(2)}%
          </div>
          <div className="font-mono text-[10px] text-[#A78BFA] mt-1">
            Raw detector output (not a probability)
          </div>
        </div>
      </div>

      {/* Meter Bar */}
      <div className="space-y-1.5">
        <div className="flex justify-between font-mono text-[11px] text-[#64748B]">
          <span>BASELINE (0.00%)</span>
          <span className="text-[#A78BFA]">DECISION THRESHOLD</span>
          <span>MAX SCORE (100.00%)</span>
        </div>
        <div className="w-full h-3 bg-white/10 rounded-full overflow-hidden p-[2px] backdrop-blur-sm">
          <div
            style={{ width: `${Math.max(2, Math.min(100, scorePct))}%` }}
            className={`h-full rounded-full transition-all duration-1000 ${
              isAI
                ? 'bg-gradient-to-r from-[#A78BFA] via-[#FF75D8] to-[#FF3366] shadow-[0_0_15px_rgba(255,117,216,0.8)]'
                : 'bg-gradient-to-r from-[#3B82F6] via-[#62E8FF] to-[#10B981] shadow-[0_0_15px_rgba(98,232,255,0.8)]'
            }`}
          />
        </div>
      </div>

      {/* Forensic Disclaimer */}
      <div className="mt-5 pt-4 border-t border-white/5 flex flex-col sm:flex-row items-start sm:items-center justify-between text-xs text-[#64748B] gap-2">
        <div className="flex items-center gap-2 text-[#94A3B8]">
          <Info className="w-4 h-4 text-[#A78BFA] shrink-0" />
          <span>
            <strong className="text-[#F5F5FF]">Methodological Note:</strong> This is an uncalibrated model score,{' '}
            <span className="text-[#FF75D8] font-semibold">not a calibrated probability</span> that the media is fake.
          </span>
        </div>
        <span className="font-mono text-[10px] text-[#62E8FF] bg-[#62E8FF]/10 border border-[#62E8FF]/20 px-2.5 py-1 rounded shrink-0">
          AUTOMATED SCREENING ONLY
        </span>
      </div>
    </div>
  );
};
