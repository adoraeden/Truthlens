import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle } from 'lucide-react';
import { Verdict } from '../types/forensics';

interface VerdictBannerProps {
  verdict: Verdict;
  score: number;
  scoreCutoff?: number;
  logit?: number;
  decisionThreshold?: number;
  subtitle?: string;
  sourceLabel?: string;
  latencyMs?: number;
  scoreLabel?: string;
  scoreScale?: 'fraction' | 'percentage';
  thresholdExplanation?: string;
}

export const VerdictBanner: React.FC<VerdictBannerProps> = ({
  verdict,
  score,
  scoreCutoff = 0.5,
  logit,
  decisionThreshold = 0.5,
  subtitle,
  sourceLabel = 'Python Neural Detector',
  latencyMs,
  scoreLabel,
  scoreScale = 'percentage',
  thresholdExplanation,
}) => {
  const isAI = verdict === 'AI-generated';
  const isMixed = verdict === 'Mixed evidence';
  const hasLogit = typeof logit === 'number' && Number.isFinite(logit);
  const safeLogit = hasLogit ? (logit as number) : null;
  const safeDecisionThreshold =
    typeof decisionThreshold === 'number' && Number.isFinite(decisionThreshold)
      ? decisionThreshold
      : 0.5;
  const logitMargin =
    safeLogit === null ? null : Math.abs(safeLogit - safeDecisionThreshold);

  const safeScoreCutoff =
    typeof scoreCutoff === 'number' && Number.isFinite(scoreCutoff)
      ? scoreCutoff
      : 0.5;
  const scoreMargin = Math.abs(score - safeScoreCutoff);

  return (
    <div
      className={`relative rounded-2xl p-6 md:p-7 border transition-all overflow-hidden ${
        isAI
          ? 'bg-gradient-to-br from-[#FF3366]/15 via-[#16182B] to-[#0E0F1D] border-[#FF3366]/60 shadow-[0_12px_40px_-15px_rgba(255,51,102,0.35)]'
          : isMixed
          ? 'bg-gradient-to-br from-[#F59E0B]/15 via-[#16182B] to-[#0E0F1D] border-[#F59E0B]/60 shadow-[0_12px_40px_-15px_rgba(245,158,11,0.35)]'
          : 'bg-gradient-to-br from-[#10B981]/15 via-[#16182B] to-[#0E0F1D] border-[#10B981]/60 shadow-[0_12px_40px_-15px_rgba(16,185,129,0.3)]'
      }`}
    >
      {/* Top glowing edge line */}
      <div
        className={`absolute top-0 left-0 right-0 h-[2px] ${
          isAI
            ? 'bg-gradient-to-r from-transparent via-[#FF75D8] to-transparent'
            : isMixed
            ? 'bg-gradient-to-r from-transparent via-[#FBBF24] to-transparent'
            : 'bg-gradient-to-r from-transparent via-[#62E8FF] to-transparent'
        }`}
      />

      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        {/* Left: Main Verdict (Visual Focus) */}
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-3">
            <span
              className={`font-mono text-[11px] font-bold px-2.5 py-0.5 rounded-md uppercase tracking-wider border ${
                isAI
                  ? 'bg-[#FF75D8]/10 text-[#FF75D8] border-[#FF75D8]/30'
                  : isMixed
                  ? 'bg-[#FBBF24]/10 text-[#FBBF24] border-[#FBBF24]/30'
                  : 'bg-[#62E8FF]/10 text-[#62E8FF] border-[#62E8FF]/30'
              }`}
            >
              Forensic Verdict
            </span>
            <span className="font-mono text-[10px] text-[#A78BFA] bg-[#A78BFA]/10 border border-[#A78BFA]/25 px-2 py-0.5 rounded">
              {sourceLabel}
            </span>
            {latencyMs !== undefined && (
              <span className="font-mono text-[10px] text-[#94A3B8] bg-white/[0.04] border border-white/10 px-2 py-0.5 rounded">
                {(latencyMs / 1000).toFixed(2)}s latency
              </span>
            )}
          </div>

          <div className="flex items-center gap-3.5">
            {isAI ? (
              <>
                <div className="w-12 h-12 rounded-xl bg-[#FF3366]/20 border border-[#FF3366]/40 flex items-center justify-center shrink-0 shadow-[0_0_20px_rgba(255,51,102,0.3)] text-[#FF75D8]">
                  <ShieldAlert className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-2xl sm:text-3xl font-extrabold text-[#FF75D8] tracking-tight">
                    AI-GENERATED
                  </h3>
                  <div className="text-[11px] font-mono text-[#FF3366] font-semibold uppercase tracking-wider">
                    Synthetic Indicators Detected Across Sampled Frames
                  </div>
                </div>
              </>
            ) : isMixed ? (
              <>
                <div className="w-12 h-12 rounded-xl bg-[#F59E0B]/20 border border-[#F59E0B]/40 flex items-center justify-center shrink-0 shadow-[0_0_20px_rgba(245,158,11,0.3)] text-[#FBBF24]">
                  <AlertTriangle className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-2xl sm:text-3xl font-extrabold text-[#FBBF24] tracking-tight">
                    MIXED EVIDENCE
                  </h3>
                  <div className="text-[11px] font-mono text-[#F59E0B] font-semibold uppercase tracking-wider">
                    Review Required — Disagreement Across Sampled Frames
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="w-12 h-12 rounded-xl bg-[#10B981]/20 border border-[#10B981]/40 flex items-center justify-center shrink-0 shadow-[0_0_20px_rgba(16,185,129,0.3)] text-[#62E8FF]">
                  <ShieldCheck className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-2xl sm:text-3xl font-extrabold text-[#62E8FF] tracking-tight">
                    LIKELY REAL
                  </h3>
                  <div className="text-[11px] font-mono text-[#10B981] font-semibold uppercase tracking-wider">
                    No Synthetic Indicators Detected in Sampled Frames
                  </div>
                </div>
              </>
            )}
          </div>

          <p className="text-xs text-[#94A3B8] mt-3 max-w-xl leading-relaxed">
            {subtitle ||
              (isAI
                ? 'Model output crossed the detection threshold based on neural latent representations.'
                : isMixed
                ? 'Sampled frames contain conflicting evidence: some frames crossed the threshold while others remained below.'
                : 'Model output remained below the synthetic classification threshold.')}
          </p>
          {thresholdExplanation && (
            <div className="mt-2 text-[11px] font-mono text-[#8290A8] bg-white/[0.03] border border-white/5 rounded-lg px-2.5 py-1.5 inline-block">
              {thresholdExplanation}
            </div>
          )}
        </div>

        {/* Right: Compact Raw Model Logit / Uncalibrated Detector Score & Decision Threshold */}
        <div className="lg:w-72 shrink-0 rounded-xl bg-[#0B0C16]/80 border border-white/10 p-3.5">
          {safeLogit !== null ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[11px]">
                <span className="font-mono uppercase tracking-wider text-[#8290A8]">Raw Model Logit</span>
                <span className="font-mono text-[10px] text-[#A78BFA]">Cutoff ≥ {safeDecisionThreshold.toFixed(3)}</span>
              </div>
              <div className={`font-mono text-3xl font-extrabold tracking-tight ${
                isAI ? 'text-[#FF75D8]' : 'text-[#62E8FF]'
              }`}>
                {safeLogit > 0 ? `+${safeLogit.toFixed(3)}` : safeLogit.toFixed(3)}
              </div>
              <div className="flex items-center justify-between pt-1 border-t border-white/5 font-mono text-[10px] text-[#94A3B8]">
                <span>Logit Threshold: {safeDecisionThreshold.toFixed(3)}</span>
                <span className={safeLogit >= safeDecisionThreshold ? 'text-[#FF75D8]' : 'text-[#62E8FF]'}>
                  Margin: {logitMargin!.toFixed(3)} {safeLogit >= safeDecisionThreshold ? 'above' : 'below'}
                </span>
              </div>
            </div>
          ) : scoreScale === 'fraction' ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[11px]">
                <span className="font-mono uppercase tracking-wider text-[#8290A8]">
                  {scoreLabel || 'Uncalibrated Detector Score'}
                </span>
                <span className="font-mono text-[10px] text-[#A78BFA]">Cutoff ≥ {safeScoreCutoff.toFixed(3)}</span>
              </div>
              <div className="text-[10px] font-mono text-[#64748B] -mt-1">
                Sigmoid-transformed • not a probability
              </div>
              <div className={`font-mono text-3xl font-extrabold tracking-tight ${
                isAI ? 'text-[#FF75D8]' : isMixed ? 'text-[#FBBF24]' : 'text-[#62E8FF]'
              }`}>
                {score.toFixed(3)}
              </div>
              <div className="flex items-center justify-between pt-1 border-t border-white/5 font-mono text-[10px] text-[#94A3B8]">
                <span>Threshold: {safeScoreCutoff.toFixed(3)}</span>
                <span className={score >= safeScoreCutoff ? (isMixed ? 'text-[#FBBF24]' : 'text-[#FF75D8]') : 'text-[#62E8FF]'}>
                  Margin: {scoreMargin.toFixed(3)} {score >= safeScoreCutoff ? 'above' : 'below'}
                </span>
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[11px]">
                <span className="font-mono uppercase tracking-wider text-[#8290A8]">
                  {scoreLabel || 'Detector Score'}
                </span>
                <span className="font-mono text-[10px] text-[#A78BFA]">Cutoff ≥ {(safeScoreCutoff * 100).toFixed(0)}%</span>
              </div>
              <div className={`font-mono text-3xl font-extrabold tracking-tight ${
                isAI ? 'text-[#FF75D8]' : 'text-[#62E8FF]'
              }`}>
                {(score * 100).toFixed(1)}%
              </div>
              <div className="flex items-center justify-between pt-1 border-t border-white/5 font-mono text-[10px] text-[#94A3B8]">
                <span>Threshold: {(safeScoreCutoff * 100).toFixed(0)}%</span>
                <span className={score >= safeScoreCutoff ? 'text-[#FF75D8]' : 'text-[#62E8FF]'}>
                  {(scoreMargin * 100).toFixed(1)}% {score >= safeScoreCutoff ? 'above' : 'below'}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
