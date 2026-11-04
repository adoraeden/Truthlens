import React, { useState, useRef, useCallback } from 'react';
import {
  Video as VideoIcon,
  Zap,
  RotateCw,
  AlertCircle,
  Film,
  Volume2,
  Clock,
  AlertTriangle,
  ChevronRight,
  RefreshCw,
} from 'lucide-react';
import { UploadZone } from './UploadZone';
import { VerdictBanner } from './VerdictBanner';
import { ScanningOverlay } from './ScanningOverlay';
import { VideoAnalysisResult, HistoryItem, BackendStatus, Verdict } from '../types/forensics';
import { analyzeVideo, fetchVideoPreview } from '../services/api';

interface VideoLabProps {
  onScanCompleted: (item: HistoryItem) => void;
  backendStatus: BackendStatus;
}

function MethodologyDetails({
  children,
  title = 'Technical Methodology & Caveats',
}: {
  children: React.ReactNode;
  title?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
      className="group rounded-xl border border-white/10 bg-[#090A14]/70 transition-colors"
    >
      <summary className="flex cursor-pointer list-none items-center justify-between p-3.5 text-xs font-medium text-[#A6B1C5] hover:text-white transition-colors select-none">
        <span className="flex items-center gap-2">
          <ChevronRight
            className={`w-3.5 h-3.5 text-[#FF75D8] transition-transform duration-200 ${
              open ? 'rotate-90' : ''
            }`}
          />
          <span className="font-mono text-[11px] uppercase tracking-wider">{title}</span>
        </span>
        <span className="font-mono text-[10px] text-[#64748B] group-hover:text-[#94A3B8]">
          {open ? 'Hide details' : 'View details'}
        </span>
      </summary>
      <div className="border-t border-white/5 p-3.5 pt-2.5 space-y-2 text-xs leading-relaxed text-[#94A3B8]">
        {children}
      </div>
    </details>
  );
}

export const VideoLab: React.FC<VideoLabProps> = ({ onScanCompleted, backendStatus }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // nativeUrl: blob URL of the original file (attempted first for native browser playback)
  // previewUrl: what the <video> element actually plays (may be converted H.264 MP4 blob URL)
  const [nativeUrl, setNativeUrl] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewState, setPreviewState] = useState<'loading' | 'native' | 'converting' | 'converted' | 'failed'>('loading');
  const [previewError, setPreviewError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const convertedUrlRef = useRef<string | null>(null);

  const [isScanning, setIsScanning] = useState(false);
  const [scanPhase, setScanPhase] = useState('');
  const [scanProgress, setScanProgress] = useState(0);
  const [result, setResult] = useState<VideoAnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Frame inspection states
  const [filterMode, setFilterMode] = useState<'all' | 'flagged'>('all');
  const [viewOverlay, setViewOverlay] = useState<Record<number, boolean>>({});

  const revokeConverted = () => {
    if (convertedUrlRef.current) {
      URL.revokeObjectURL(convertedUrlRef.current);
      convertedUrlRef.current = null;
    }
  };

  const requestConvertedPreview = useCallback(async (file: File) => {
    if (!backendStatus.connected) {
      setPreviewState('failed');
      setPreviewError('Video preview unavailable: backend is offline. The file will still be submitted for forensic analysis.');
      return;
    }
    setPreviewState('converting');
    try {
      const convertedUrl = await fetchVideoPreview(file);
      revokeConverted();
      convertedUrlRef.current = convertedUrl;
      setPreviewUrl(convertedUrl);
      setPreviewState('converted');
      setPreviewError(null);
    } catch (err: unknown) {
      setPreviewState('failed');
      setPreviewError(
        `Browser cannot decode this video codec/container natively, and preview conversion failed: ${
          err instanceof Error ? err.message : String(err)
        }. The file will still be submitted for forensic analysis.`
      );
    }
  }, [backendStatus.connected]);

  const handleFileSelected = (file: File) => {
    if (nativeUrl) URL.revokeObjectURL(nativeUrl);
    revokeConverted();

    setSelectedFile(file);
    setResult(null);
    setErrorMessage(null);
    setFilterMode('all');
    setViewOverlay({});
    setPreviewState('loading');
    setPreviewError(null);

    const url = URL.createObjectURL(file);
    setNativeUrl(url);
    setPreviewUrl(url);
  };

  const clearFile = () => {
    if (nativeUrl) URL.revokeObjectURL(nativeUrl);
    revokeConverted();

    setSelectedFile(null);
    setNativeUrl(null);
    setPreviewUrl(null);
    setResult(null);
    setErrorMessage(null);
    setFilterMode('all');
    setViewOverlay({});
    setPreviewState('loading');
    setPreviewError(null);
  };

  const handleLoadedMetadata = () => {
    setPreviewState((prev) => (prev === 'loading' ? 'native' : prev));
  };

  const handleVideoError = () => {
    if (previewState === 'loading' || previewState === 'native') {
      if (selectedFile) {
        requestConvertedPreview(selectedFile);
      }
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;

    if (!backendStatus.connected) {
      setErrorMessage(
        "The Python detector backend is offline. From the backend folder, run .\\.venv\\Scripts\\python.exe -m uvicorn src.api_server:app --host 127.0.0.1 --port 8002."
      );
      return;
    }

    setIsScanning(true);
    setResult(null);
    setErrorMessage(null);
    setScanProgress(15);
    setScanPhase('Reading video duration via FFprobe...');

    const timer1 = setTimeout(() => {
      setScanProgress(40);
      setScanPhase('Demuxing audio track with FFmpeg (16kHz mono PCM)...');
    }, 500);

    const timer2 = setTimeout(() => {
      setScanProgress(70);
      setScanPhase('Extracting uniform keyframe slices (up to 8 frames)...');
    }, 1100);

    const timer3 = setTimeout(() => {
      setScanProgress(90);
      setScanPhase(
        backendStatus.models?.audio === false
          ? 'Evaluating visual frames; PellaV2 audio analysis is unavailable...'
          : 'Evaluating keyframes with ViT and audio with PellaV2...'
      );
    }, 1700);

    try {
      const analysis = await analyzeVideo(selectedFile);
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      setScanProgress(100);
      setIsScanning(false);
      setResult(analysis);

      // Record in session history: 3-state video classification
      const flaggedFrames = analysis.ai_flagged_frames;
      const totalFrames = analysis.frames_analyzed;
      const allFramesAi = totalFrames > 0 && flaggedFrames === totalFrames;
      const someFramesAi = flaggedFrames > 0 && flaggedFrames < totalFrames;
      const zeroFramesAi = flaggedFrames === 0;

      let historyVerdictDisplay: string;
      if (allFramesAi) {
        historyVerdictDisplay = 'AI-GENERATED (across all sampled frames)';
      } else if (someFramesAi) {
        historyVerdictDisplay = `MIXED EVIDENCE (${flaggedFrames}/${totalFrames} frames flagged)`;
      } else {
        historyVerdictDisplay = 'LIKELY REAL (no synthetic indicators in sampled frames)';
      }

      const historyItem: HistoryItem = {
        id: `vid-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }),
        fileName: selectedFile.name,
        mediaType: 'video',
        fileSizeFormatted: `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB`,
        verdictDisplay: historyVerdictDisplay,
        scoreDisplay: `Mean frame score: ${analysis.average_frame_score.toFixed(3)}${
          analysis.audio_analysis
            ? ` • Audio score: ${analysis.audio_analysis.score.toFixed(3)}`
            : ''
        }`,
        details: `${analysis.frames_analyzed} frames • ${analysis.ai_flagged_frames} flagged • ${analysis.duration_seconds.toFixed(1)}s`,
        latencyMs: analysis.inferenceDurationMs,
        fullResult: { type: 'video', data: analysis },
      };
      onScanCompleted(historyItem);
    } catch (err: any) {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      setIsScanning(false);
      setErrorMessage(err.message || 'Video analysis failed.');
    }
  };

  // Modality agreement & verdict flags
  const visualFlaggedCount = result?.ai_flagged_frames || 0;
  const totalFramesCount = result?.frames_analyzed || 0;
  const allFramesFlagged = totalFramesCount > 0 && visualFlaggedCount === totalFramesCount;
  const someFramesFlagged = visualFlaggedCount > 0 && visualFlaggedCount < totalFramesCount;
  const zeroFramesFlagged = visualFlaggedCount === 0;

  const audioFlagged = result?.audio_analysis?.verdict === 'AI-generated';
  const modalitiesDisagree =
    result && result.audio_analysis !== null && (visualFlaggedCount > 0) !== audioFlagged;

  // Conservative 3-state video verdict presentation:
  // 1. Zero flagged frames -> Likely real
  // 2. Some but not all frames flagged -> Mixed evidence
  // 3. All frames flagged -> AI-generated
  let videoVerdict: Verdict;
  if (allFramesFlagged) {
    videoVerdict = 'AI-generated';
  } else if (someFramesFlagged) {
    videoVerdict = 'Mixed evidence';
  } else {
    videoVerdict = 'Likely real';
  }

  const filteredFrames =
    result?.frame_results.filter((f) =>
      filterMode === 'flagged' ? f.verdict === 'AI-generated' : true
    ) || [];

  return (
    <div className="space-y-6">
      {/* Workspace Header */}
      <div className="bg-[#191B30]/80 border border-[#FF75D8]/30 rounded-2xl p-6 backdrop-blur-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-white/10 gap-3">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-[#FF75D8]/10 border border-[#FF75D8]/30 flex items-center justify-center text-[#FF75D8]">
                <VideoIcon className="w-5 h-5" />
              </div>
              <h3 className="text-xl font-bold tracking-tight text-[#FF75D8]">
                VIDEO FORENSIC LAB
              </h3>
            </div>
            <p className="text-xs md:text-sm text-[#94A3B8] mt-1.5">
              Multi-frame temporal screening and dual-modality forensics across sampled video frames and demuxed audio.
            </p>
          </div>

          <span className="font-mono text-xs font-bold px-3 py-1.5 rounded-lg border border-[#FF75D8]/30 bg-[#FF75D8]/10 text-[#FF75D8] self-start sm:self-auto">
            FORMATS: MP4, MOV, MKV, AVI
          </span>
        </div>

        {/* Upload Zone */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
          <div className="lg:col-span-2">
            <UploadZone
              onFileSelected={handleFileSelected}
              acceptedFormats={['.MP4', '.MOV', '.MKV', '.AVI']}
              acceptedMimeTypes="video/*"
              mediaLabel="video container"
              selectedFile={selectedFile}
              onClearFile={clearFile}
              accentColor="pink"
            />
          </div>

          {/* Action Card */}
          <div className="flex flex-col justify-between bg-[#121424] border border-white/5 rounded-2xl p-5">
            <div>
              <div className="font-mono text-xs text-[#FF75D8] font-bold uppercase tracking-wider mb-2">
                Dual-Modality Protocol
              </div>
              <p className="text-xs text-[#94A3B8] leading-relaxed mb-4">
                Video frames (ViT-Small) and demuxed audio tracks (PellaV2) are evaluated independently without artificial multimodal score blending.
              </p>

              <div className="p-3 rounded-xl bg-[#10111F] border border-white/5 font-mono text-[11px] text-[#94A3B8] space-y-1.5">
                <div className="flex justify-between">
                  <span>MAX FRAMES:</span>
                  <span className="text-white">Up to 8 frames</span>
                </div>
                <div className="flex justify-between">
                  <span>SAMPLING:</span>
                  <span className="text-white">Interval Midpoints</span>
                </div>
                <div className="flex justify-between">
                  <span>AUDIO TRACK:</span>
                  <span className="text-[#FF75D8]">Demuxed 16kHz</span>
                </div>
              </div>
            </div>

            <button
              disabled={!selectedFile || isScanning}
              onClick={runAnalysis}
              className="mt-4 w-full py-3 px-4 rounded-xl font-bold text-xs tracking-wider uppercase text-white bg-gradient-to-r from-[#FF0080] via-[#7928CA] to-[#00D2FF] border border-white/20 hover:shadow-[0_0_25px_rgba(255,117,216,0.4)] transition-all disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer flex items-center justify-center gap-2"
            >
              {isScanning ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" />
                  <span>ANALYZING VIDEO...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>ANALYSE VIDEO ⚡</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Backend Error Alert if any */}
        {errorMessage && (
          <div className="mt-5 p-4 rounded-xl bg-[#FF3366]/15 border border-[#FF3366]/40 flex items-start gap-3 text-xs text-[#FF75D8]">
            <AlertCircle className="w-5 h-5 text-[#FF3366] shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">Execution Notice:</strong> {errorMessage}
            </div>
          </div>
        )}

        {/* Video Player Preview */}
        {previewUrl && (
          <div className="mt-6 pt-6 border-t border-white/10 space-y-3">
            <div className="flex items-center justify-between font-mono text-xs text-[#FF75D8]">
              <span>LOADED VIDEO CONTAINER</span>
              <span className="text-[#94A3B8]">{selectedFile?.name}</span>
            </div>

            {/* Preview state: converting indicator */}
            {previewState === 'converting' && (
              <div className="flex items-center gap-2 text-xs text-[#FF75D8] bg-[#FF75D8]/10 border border-[#FF75D8]/30 rounded-xl px-3 py-2">
                <RefreshCw className="w-3.5 h-3.5 animate-spin shrink-0" />
                <span>Generating browser-compatible preview (H.264 MP4) via FFmpeg — forensic analysis will use the original file.</span>
              </div>
            )}

            {/* Preview state: failed indicator */}
            {previewState === 'failed' && previewError && (
              <div className="flex items-start gap-2 text-xs text-[#FF75D8] bg-[#FF3366]/10 border border-[#FF3366]/30 rounded-xl px-3 py-2">
                <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5 text-[#FF3366]" />
                <span>{previewError}</span>
              </div>
            )}

            {/* Preview state: converted badge */}
            {previewState === 'converted' && (
              <div className="flex items-center gap-2 text-xs text-[#10B981] bg-[#10B981]/10 border border-[#10B981]/30 rounded-xl px-3 py-2">
                <Film className="w-3.5 h-3.5 shrink-0" />
                <span>Preview converted to H.264 MP4 for browser playback. Original uploaded file submitted for forensic analysis.</span>
              </div>
            )}

            <div className="relative rounded-2xl bg-[#0E0F1C] border border-[#FF75D8]/30 p-3 overflow-hidden max-w-2xl mx-auto shadow-xl">
              <video
                ref={videoRef}
                src={previewUrl}
                controls
                playsInline
                onLoadedMetadata={handleLoadedMetadata}
                onError={handleVideoError}
                className="w-full rounded-xl max-h-72 object-contain bg-black"
              />
            </div>
          </div>
        )}

        {/* Scanning Animation */}
        {isScanning && (
          <ScanningOverlay
            currentPhase={scanPhase}
            progressPercent={scanProgress}
            mediaType="video"
            modelLabel={
              backendStatus.models?.audio === false
                ? 'ViT-Small/16 (Frames) · Audio detector unavailable'
                : 'ViT-Small/16 (Frames) + PellaV2 (Demuxed Audio)'
            }
          />
        )}

        {/* RESULTS SECTION */}
        {result && (
          <div className="mt-8 space-y-4 animate-in fade-in duration-500">
            {/* 1. Main Video Verdict Banner (Primary Visual Focus) */}
            <VerdictBanner
              verdict={videoVerdict}
              score={result.average_frame_score}
              scoreCutoff={0.5}
              scoreLabel="Mean Uncalibrated Frame Score"
              scoreScale="fraction"
              sourceLabel="Dual-Modality (ViT + PellaV2)"
              latencyMs={result.inferenceDurationMs}
              thresholdExplanation="Threshold Rule: Individual frame score at or above 0.500 → AI-generated frame; below 0.500 → Likely Real frame. The 0.500 cutoff applies to each frame independently, not to the mean frame score as a whole-video probability."
              subtitle={
                allFramesFlagged
                  ? `All ${result.frames_analyzed} sampled frames crossed the detection threshold (score ≥ 0.500)${
                      audioFlagged ? ' • Demuxed audio track also flagged as synthetic' : ''
                    }.`
                  : someFramesFlagged
                  ? `${result.ai_flagged_frames} of ${result.frames_analyzed} sampled frames flagged (score ≥ 0.500). Isolated flagged frames do not constitute whole-video AI generation; manual review of highlighted frames is required${
                      audioFlagged ? ' • Demuxed audio track flagged as synthetic' : ''
                    }.`
                  : `0 of ${result.frames_analyzed} sampled frames flagged (all frame scores < 0.500); no synthetic indicators detected in sampled frames${
                      result.audio_analysis ? (audioFlagged ? ' • Audio track flagged as synthetic' : ' • Audio track evaluated as likely authentic') : ''
                    }.`
              }
            />

            {/* Modality Disagreement Alert Banner */}
            {modalitiesDisagree && (
              <div className="p-4 rounded-xl bg-[#FF75D8]/10 border border-[#FF75D8]/30 flex items-start gap-3 text-xs text-[#F5F5FF]">
                <AlertTriangle className="w-5 h-5 text-[#FF75D8] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[#FF75D8] uppercase tracking-wide">
                    Modality Disagreement:
                  </strong>{' '}
                  Visual frames and audio track yielded differing verdicts. Visual frames flagged{' '}
                  <span className="font-bold text-[#FF75D8]">{result.ai_flagged_frames} frame(s)</span>, while
                  audio was evaluated as{' '}
                  <span className="font-bold text-[#62E8FF]">{result.audio_analysis?.verdict}</span>. Modalities are evaluated separately.
                </div>
              </div>
            )}

            {/* 2. Compact Telemetry Strip */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-[#121424] border border-white/5 rounded-xl p-3">
                <div className="font-mono text-[10px] text-[#8290A8] uppercase tracking-wider">
                  Video Duration
                </div>
                <div className="font-mono text-base font-bold text-[#62E8FF] mt-0.5">
                  {result.duration_seconds.toFixed(1)}s
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3">
                <div className="font-mono text-[10px] text-[#8290A8] uppercase tracking-wider">
                  Sampled Frames
                </div>
                <div className="font-mono text-base font-bold text-white mt-0.5">
                  {result.frames_analyzed} Frames
                  {result.ai_flagged_frames > 0 && (
                    <span className="text-xs text-[#FF75D8] font-normal ml-1">
                      ({result.ai_flagged_frames} flagged)
                    </span>
                  )}
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3">
                <div className="font-mono text-[10px] text-[#8290A8] uppercase tracking-wider">
                  Mean Frame Score
                </div>
                <div className="font-mono text-base font-bold text-[#A78BFA] mt-0.5">
                  {result.average_frame_score.toFixed(3)}
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3">
                <div className="font-mono text-[10px] text-[#8290A8] uppercase tracking-wider">
                  Audio Modality
                </div>
                <div className="font-mono text-base font-bold text-[#10B981] mt-0.5 truncate">
                  {result.audio_analysis
                    ? result.audio_analysis.verdict
                    : result.audio_track_present
                    ? 'Stream Unreadable'
                    : 'No Audio Track'}
                </div>
              </div>
            </div>

            {/* 3. Probability Calibration Status (Strictly uncalibrated, no probabilities invented) */}
            <section
              className="rounded-xl border border-white/10 bg-[#121424]/90 p-4"
              aria-label="Probability calibration"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs uppercase tracking-wider text-[#FF75D8] font-semibold">
                    Probability Calibration
                  </span>
                  <span className="rounded-full border border-white/15 bg-white/[0.04] px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-[#A6B1C5]">
                    Unavailable
                  </span>
                </div>
                <span className="text-[11px] text-[#8290A8]">
                  Independent from threshold verdict
                </span>
              </div>

              <p className="mt-2 text-xs text-[#94A3B8] leading-relaxed">
                Calibrated probability is currently unavailable. TruthLens reports raw uncalibrated detector scores for each sampled frame and audio track; whole-video probabilities are withheld until independent multi-frame calibration is validated.
              </p>
            </section>

            {/* 4. Sampled Video Frames Timeline & Inspection Grid */}
            <div className="rounded-2xl border border-white/10 bg-[#0E0F1C]/90 p-5 space-y-4">
              {/* Header with filter controls */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
                <div className="flex items-center gap-2.5">
                  <Film className="w-4 h-4 text-[#FF75D8]" />
                  <h4 className="font-bold text-sm text-white tracking-tight">
                    SAMPLED VIDEO FRAMES ({result.frames_analyzed} EXTRACTED)
                  </h4>
                </div>

                <div className="flex items-center gap-2 self-start sm:self-auto">
                  <button
                    type="button"
                    onClick={() => setFilterMode('all')}
                    className={`px-2.5 py-1 rounded-lg font-mono text-[11px] font-semibold transition cursor-pointer ${
                      filterMode === 'all'
                        ? 'bg-white/10 text-white border border-white/20'
                        : 'text-[#8290A8] hover:text-white'
                    }`}
                  >
                    All ({result.frames_analyzed})
                  </button>
                  {result.ai_flagged_frames > 0 && (
                    <button
                      type="button"
                      onClick={() => setFilterMode('flagged')}
                      className={`px-2.5 py-1 rounded-lg font-mono text-[11px] font-semibold transition flex items-center gap-1.5 cursor-pointer ${
                        filterMode === 'flagged'
                          ? 'bg-[#FF3366]/20 text-[#FF75D8] border border-[#FF3366]/40'
                          : 'text-[#FF75D8]/70 hover:text-[#FF75D8]'
                      }`}
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-[#FF3366]" />
                      Flagged ({result.ai_flagged_frames})
                    </button>
                  )}
                </div>
              </div>

              {/* Visual Temporal Timeline Bar */}
              <div className="space-y-1.5">
                <div className="flex justify-between font-mono text-[10px] text-[#8290A8]">
                  <span>0.0s (Start)</span>
                  <span>Temporal Interval Sampling</span>
                  <span>{result.duration_seconds.toFixed(1)}s (End)</span>
                </div>
                <div className="relative h-3.5 w-full bg-[#07080F] rounded-full border border-white/10 overflow-hidden">
                  {result.frame_results.map((frame, idx) => {
                    const isFrameAi = frame.verdict === 'AI-generated';
                    const pct = Math.max(
                      0,
                      Math.min(100, (frame.timestamp_seconds / result.duration_seconds) * 100)
                    );
                    return (
                      <div
                        key={idx}
                        className={`absolute top-0 bottom-0 w-2.5 -ml-1 rounded-sm transition-transform hover:scale-125 ${
                          isFrameAi
                            ? 'bg-[#FF3366] shadow-[0_0_6px_#FF3366]'
                            : 'bg-[#62E8FF] shadow-[0_0_4px_#62E8FF]'
                        }`}
                        style={{ left: `${pct}%` }}
                        title={`Frame ${idx + 1} at ${frame.timestamp_seconds.toFixed(2)}s: ${frame.verdict}`}
                      />
                    );
                  })}
                </div>
              </div>

              {/* Frame Cards Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
                {filteredFrames.map((frame, idx) => {
                  const isFrameAi = frame.verdict === 'AI-generated';
                  const showOverlay = viewOverlay[idx];
                  const hasImages =
                    Boolean(frame.explanation?.crop_png_base64) &&
                    Boolean(frame.explanation?.overlay_png_base64);

                  return (
                    <div
                      key={idx}
                      className={`p-3.5 rounded-xl border transition-all ${
                        isFrameAi
                          ? 'bg-[#FF3366]/10 border-[#FF3366]/40 shadow-[0_4px_20px_-10px_rgba(255,51,102,0.3)]'
                          : 'bg-[#10111F]/80 border-white/10 hover:border-white/20'
                      }`}
                    >
                      {/* Frame Header: Timestamp & Verdict */}
                      <div className="flex items-center justify-between mb-2.5">
                        <span className="font-mono text-[11px] text-[#A6B1C5] font-semibold flex items-center gap-1.5">
                          <Clock className="w-3 h-3 text-[#FF75D8]" />
                          <span>{frame.timestamp_seconds.toFixed(2)}s</span>
                        </span>
                        <span
                          className={`font-mono text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
                            isFrameAi
                              ? 'bg-[#FF75D8]/20 text-[#FF75D8] border border-[#FF75D8]/40'
                              : 'bg-[#62E8FF]/20 text-[#62E8FF] border border-[#62E8FF]/40'
                          }`}
                        >
                          {isFrameAi ? 'AI-Flagged' : 'Likely Real'}
                        </span>
                      </div>

                      {/* Image Thumbnail / Attribution Preview */}
                      {hasImages && frame.explanation && (
                        <div className="relative aspect-square w-full rounded-lg overflow-hidden bg-black/40 border border-white/10 mb-2.5">
                          <img
                            src={`data:image/png;base64,${
                              showOverlay
                                ? frame.explanation.overlay_png_base64
                                : frame.explanation.crop_png_base64
                            }`}
                            alt={`Frame at ${frame.timestamp_seconds}s model input`}
                            className="h-full w-full object-cover"
                          />
                          <button
                            type="button"
                            onClick={() =>
                              setViewOverlay((prev) => ({ ...prev, [idx]: !prev[idx] }))
                            }
                            className="absolute bottom-1.5 right-1.5 px-2 py-0.5 rounded font-mono text-[9px] font-semibold bg-black/80 text-white border border-white/20 hover:border-white/50 transition cursor-pointer backdrop-blur-sm"
                          >
                            {showOverlay ? 'Crop' : 'Overlay'}
                          </button>
                        </div>
                      )}

                      {/* Frame Score & Metrics */}
                      <div className="space-y-1.5 font-mono text-xs">
                        <div className="flex justify-between items-center text-[11px]">
                          <span className="text-[#8290A8]">Frame Score:</span>
                          <span
                            className={`font-bold ${
                              isFrameAi ? 'text-[#FF75D8]' : 'text-[#62E8FF]'
                            }`}
                          >
                            {frame.score.toFixed(3)}
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                          <div
                            style={{ width: `${Math.max(2, frame.score * 100)}%` }}
                            className={`h-full ${
                              isFrameAi ? 'bg-[#FF75D8]' : 'bg-[#62E8FF]'
                            }`}
                          />
                        </div>
                        <div className="flex justify-between items-center text-[10px] text-[#8290A8] pt-0.5">
                          <span>
                            Logit: <strong className="text-white">{frame.logit > 0 ? `+${frame.logit.toFixed(3)}` : frame.logit.toFixed(3)}</strong>
                          </span>
                          <span>Cutoff: ≥ {frame.decision_threshold.toFixed(3)}</span>
                        </div>
                        <div className="text-[10px] text-[#8290A8] flex justify-between">
                          <span>Margin: {Math.abs(frame.logit - frame.decision_threshold).toFixed(3)}</span>
                          <span className={frame.logit >= frame.decision_threshold ? 'text-[#FF75D8]' : 'text-[#62E8FF]'}>
                            {frame.logit >= frame.decision_threshold ? 'above' : 'below'}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 5. Separate Audio Track Analysis Panel */}
            <div className="rounded-2xl bg-[#0E0F1C]/90 border border-[#A78BFA]/25 p-5 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-white/10 gap-2">
                <div className="flex items-center gap-2">
                  <Volume2 className="w-4 h-4 text-[#A78BFA]" />
                  <h4 className="font-bold text-sm text-[#A78BFA] uppercase tracking-wide">
                    Demuxed Audio Track Analysis (Separate Modality)
                  </h4>
                </div>

                {result.audio_analysis ? (
                  <span
                    className={`font-mono text-xs font-bold px-2.5 py-0.5 rounded uppercase self-start sm:self-auto ${
                      result.audio_analysis.verdict === 'AI-generated'
                        ? 'bg-[#FF75D8]/20 text-[#FF75D8] border border-[#FF75D8]/40'
                        : 'bg-[#62E8FF]/20 text-[#62E8FF] border border-[#62E8FF]/40'
                    }`}
                  >
                    {result.audio_analysis.verdict}
                  </span>
                ) : (
                  <span className="font-mono text-[11px] text-[#8290A8] self-start sm:self-auto">
                    {result.audio_status === 'audio_detector_unavailable'
                      ? 'Audio Detector Unavailable'
                      : result.audio_track_present
                      ? 'Audio Stream Unreadable'
                      : 'No Audio Stream Found'}
                  </span>
                )}
              </div>

              {result.audio_analysis ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="bg-[#121424] p-3 rounded-xl border border-white/5 font-mono text-xs">
                      <div className="text-[10px] uppercase tracking-wider text-[#8290A8]">
                        Uncalibrated Audio Score
                      </div>
                      <div className="text-lg font-bold text-white mt-0.5">
                        {result.audio_analysis.score.toFixed(3)}
                      </div>
                      <div className="text-[10px] text-[#8290A8]">Cutoff ≥ 0.500 (not a probability)</div>
                    </div>

                    <div className="bg-[#121424] p-3 rounded-xl border border-white/5 font-mono text-xs">
                      <div className="text-[10px] uppercase tracking-wider text-[#8290A8]">
                        Window Analyzed
                      </div>
                      <div className="text-lg font-bold text-[#62E8FF] mt-0.5">
                        {result.audio_analysis.duration_analyzed_seconds.toFixed(1)}s
                      </div>
                      <div className="text-[10px] text-[#8290A8]">
                        {result.audio_analysis.analysis_region === 'center_crop'
                          ? `Center window starts at ${
                              result.audio_analysis.analysis_start_seconds?.toFixed(1) ?? '0.0'
                            }s`
                          : 'Full short audio clip'}
                      </div>
                    </div>

                    <div className="bg-[#121424] p-3 rounded-xl border border-white/5 font-mono text-xs">
                      <div className="text-[10px] uppercase tracking-wider text-[#8290A8]">
                        Audio Preprocessing
                      </div>
                      <div className="text-lg font-bold text-[#A78BFA] mt-0.5">16,000 Hz</div>
                      <div className="text-[10px] text-[#8290A8]">1-channel Mono PCM</div>
                    </div>
                  </div>

                  <p className="text-xs text-[#94A3B8] leading-relaxed">
                    <strong className="text-white">Acoustic Note:</strong> {result.audio_analysis.note}
                  </p>
                </div>
              ) : (
                <p className="text-xs text-[#94A3B8] leading-relaxed">
                  {result.audio_status === 'audio_detector_unavailable'
                    ? result.audio_status_note ||
                      'The audio detector was not initialized; no conclusion was made about the audio.'
                    : result.audio_track_present
                    ? 'An audio stream is present but could not be decoded for analysis. The sampled visual frame results are shown separately.'
                    : 'No audio stream was found. The sampled visual frame results are shown separately.'}
                </p>
              )}
            </div>

            {/* 6. Expandable Video Methodology & Technical Caveats */}
            <MethodologyDetails title="Video Forensics Methodology & Temporal Caveats">
              <p>
                <strong className="text-white">Dual-Modality Processing:</strong> Sampled video frames and extracted audio tracks are evaluated by independent detector pipelines. Visual frames are processed by ViT-Small/16, while demuxed audio is evaluated by PellaV2. Modality scores are never combined into a single fabricated score.
              </p>
              <p>
                <strong className="text-white">Temporal Frame Sampling:</strong> {result.frame_sampling_method || 'Uniform temporal midpoint sampling.'} Because only up to {result.frames_requested ?? 8} frames are evaluated, transient deepfake manipulations occurring between sampled frames may not be captured.
              </p>
              <p>
                <strong className="text-white">Evaluation Rule:</strong> {result.interpretation}
              </p>
              <p className="text-[11px] text-[#8290A8]">
                Frame scores and audio scores are uncalibrated detector indices, not probabilities. Real videos may produce isolated false positives due to compression or motion blur; replayed or re-encoded synthetic content may evade detection.
              </p>
            </MethodologyDetails>
          </div>
        )}
      </div>
    </div>
  );
};
