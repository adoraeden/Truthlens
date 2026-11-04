import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Mic,
  Zap,
  RotateCw,
  AlertCircle,
  Play,
  Pause,
  ChevronRight,
  Activity,
  Volume2,
  RefreshCw,
} from 'lucide-react';
import { UploadZone } from './UploadZone';
import { VerdictBanner } from './VerdictBanner';
import { ScanningOverlay } from './ScanningOverlay';
import { AudioAnalysisResult, HistoryItem, BackendStatus } from '../types/forensics';
import { analyzeAudio, fetchAudioPreview } from '../services/api';

interface AudioLabProps {
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
            className={`w-3.5 h-3.5 text-[#A78BFA] transition-transform duration-200 ${
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

export const AudioLab: React.FC<AudioLabProps> = ({ onScanCompleted, backendStatus }) => {
  const audioDetectorAvailable = backendStatus.models?.audio !== false;
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // nativeUrl: blob URL of the original file (used for waveform decoding attempt)
  // previewUrl: what the <audio> element actually plays (may be a converted MP3 blob URL)
  const [nativeUrl, setNativeUrl] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // 'loading'    → waiting for <audio> onLoadedMetadata or onError
  // 'native'     → browser decoded the original file fine
  // 'converting' → native failed, requesting converted MP3 from backend
  // 'converted'  → converted MP3 is ready and loaded in <audio>
  // 'failed'     → conversion also failed
  const [previewState, setPreviewState] = useState<'loading' | 'native' | 'converting' | 'converted' | 'failed'>('loading');
  const [previewError, setPreviewError] = useState<string | null>(null);

  const [isScanning, setIsScanning] = useState(false);
  const [scanPhase, setScanPhase] = useState('');
  const [scanProgress, setScanProgress] = useState(0);
  const [result, setResult] = useState<AudioAnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Audio player & waveform states
  const [waveformPeaks, setWaveformPeaks] = useState<number[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Keep a ref to the converted preview URL so we can revoke it on cleanup
  const convertedUrlRef = useRef<string | null>(null);

  const revokeConverted = () => {
    if (convertedUrlRef.current) {
      URL.revokeObjectURL(convertedUrlRef.current);
      convertedUrlRef.current = null;
    }
  };

  const handleFileSelected = (file: File) => {
    // Revoke any previous URLs
    if (nativeUrl) URL.revokeObjectURL(nativeUrl);
    revokeConverted();

    setSelectedFile(file);
    setResult(null);
    setErrorMessage(null);
    setIsPlaying(false);
    setCurrentTime(0);
    setAudioDuration(0);
    setPreviewState('loading');
    setPreviewError(null);
    setWaveformPeaks([]);

    const url = URL.createObjectURL(file);
    setNativeUrl(url);
    setPreviewUrl(url);  // start with native; may be replaced on error
  };

  const clearFile = () => {
    if (nativeUrl) URL.revokeObjectURL(nativeUrl);
    revokeConverted();
    setSelectedFile(null);
    setNativeUrl(null);
    setPreviewUrl(null);
    setResult(null);
    setErrorMessage(null);
    setWaveformPeaks([]);
    setIsPlaying(false);
    setCurrentTime(0);
    setAudioDuration(0);
    setPreviewState('loading');
    setPreviewError(null);
  };

  // ── Fallback: ask the backend to convert to MP3 ───────────────────────────
  const requestConvertedPreview = useCallback(async (file: File) => {
    if (!backendStatus.connected) {
      setPreviewState('failed');
      setPreviewError('Audio preview unavailable: backend is offline. The file will still be submitted for forensic analysis.');
      return;
    }
    setPreviewState('converting');
    try {
      const convertedUrl = await fetchAudioPreview(file);
      revokeConverted();
      convertedUrlRef.current = convertedUrl;
      setPreviewUrl(convertedUrl);
      setPreviewState('converted');
      setPreviewError(null);
    } catch (err: unknown) {
      setPreviewState('failed');
      setPreviewError(
        `Browser cannot play this format natively, and preview conversion failed: ${
          err instanceof Error ? err.message : String(err)
        }. The file will still be submitted for forensic analysis.`
      );
    }
  }, [backendStatus.connected]);

  // ── <audio> event handlers ────────────────────────────────────────────────
  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      const dur = audioRef.current.duration;
      if (Number.isFinite(dur) && dur > 0) {
        setAudioDuration(dur);
      }
    }
    // Mark native as working only if we haven't already fallen back
    setPreviewState((prev) => (prev === 'loading' ? 'native' : prev));
  };

  const handleAudioError = () => {
    // Only trigger fallback if we were still on the native (original) URL
    if (previewState === 'loading' || previewState === 'native') {
      if (selectedFile) {
        requestConvertedPreview(selectedFile);
      }
    }
  };

  // Decode audio in browser to generate visual waveform peaks
  useEffect(() => {
    if (!selectedFile) {
      setWaveformPeaks([]);
      return;
    }

    let isCancelled = false;

    const generateWaveform = async () => {
      try {
        const arrayBuffer = await selectedFile.arrayBuffer();
        const AudioCtx =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        if (!AudioCtx) {
          throw new Error('Web Audio not supported');
        }
        const ctx = new AudioCtx();
        const decoded = await ctx.decodeAudioData(arrayBuffer);
        if (isCancelled) return;

        // Browser successfully decoded the file — set duration from it
        setAudioDuration(decoded.duration);
        // If the <audio> element had already errored, the decoding succeeded
        // so the waveform is real; keep whichever previewState is set.

        const rawData = decoded.getChannelData(0);
        const sampleBins = 100;
        const blockSize = Math.floor(rawData.length / sampleBins);
        const peaks: number[] = [];

        for (let i = 0; i < sampleBins; i++) {
          let sum = 0;
          for (let j = 0; j < blockSize; j++) {
            sum += Math.abs(rawData[i * blockSize + j]);
          }
          peaks.push(sum / blockSize);
        }

        const maxPeak = Math.max(...peaks, 0.001);
        const normalized = peaks.map((p) => Math.max(0.08, p / maxPeak));

        if (!isCancelled) {
          setWaveformPeaks(normalized);
        }
        ctx.close();
      } catch {
        if (!isCancelled) {
          // Provide a visual envelope fallback for the waveform bars while the
          // actual duration will come from the <audio> element (native or converted).
          const fallback = Array.from({ length: 100 }, (_, i) => {
            const envelope = Math.sin((i / 100) * Math.PI);
            return Math.max(0.12, envelope * (0.4 + 0.6 * ((i * 17) % 10) / 10));
          });
          setWaveformPeaks(fallback);
        }
      }
    };

    generateWaveform();

    return () => {
      isCancelled = true;
    };
  }, [selectedFile]);

  // Audio player time updates
  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  // NOTE: handleLoadedMetadata and handleAudioError are defined above (lines ~162–180)
  // and are reused on the <audio> element. No duplicate definition here.

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const handleWaveformClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !audioDuration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const progress = Math.max(0, Math.min(1, clickX / rect.width));
    const targetTime = progress * audioDuration;
    audioRef.current.currentTime = targetTime;
    setCurrentTime(targetTime);
  };

  const formatSeconds = (sec: number) => {
    if (!Number.isFinite(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;

    if (!backendStatus.connected) {
      setErrorMessage(
        "The Python detector backend is offline. From the backend folder, run .\\.venv\\Scripts\\python.exe -m uvicorn src.api_server:app --host 127.0.0.1 --port 8002."
      );
      return;
    }
    if (!audioDetectorAvailable) {
      setErrorMessage(
        backendStatus.audioUnavailableReason || 'PellaV2 is unavailable under the current memory limit.'
      );
      return;
    }

    setIsScanning(true);
    setResult(null);
    setErrorMessage(null);
    setScanProgress(20);
    setScanPhase('Demuxing and converting input to 16,000 Hz mono PCM via FFmpeg...');

    const timer1 = setTimeout(() => {
      setScanProgress(55);
      setScanPhase('Extracting 4-second temporal evaluation window...');
    }, 450);

    const timer2 = setTimeout(() => {
      setScanProgress(85);
      setScanPhase('Evaluating neural representations with PellaV2 (Wav2Vec-BERT)...');
    }, 950);

    try {
      const analysis = await analyzeAudio(selectedFile);
      clearTimeout(timer1);
      clearTimeout(timer2);
      setScanProgress(100);
      setIsScanning(false);
      setResult(analysis);

      // Record in session history
      const historyItem: HistoryItem = {
        id: `aud-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }),
        fileName: selectedFile.name,
        mediaType: 'audio',
        fileSizeFormatted: `${(selectedFile.size / 1024).toFixed(1)} KB`,
        verdictDisplay: analysis.verdict,
        scoreDisplay: `Uncalibrated score ${analysis.score.toFixed(3)}`,
        details: `${analysis.duration_analyzed_seconds.toFixed(1)}s analyzed • ${
          analysis.analysis_region === 'center_crop' ? '4s center window' : 'full clip'
        } • 16kHz mono`,
        latencyMs: analysis.inferenceDurationMs,
        fullResult: { type: 'audio', data: analysis },
      };
      onScanCompleted(historyItem);
    } catch (err: any) {
      clearTimeout(timer1);
      clearTimeout(timer2);
      setIsScanning(false);
      setErrorMessage(err.message || 'Audio analysis failed.');
    }
  };

  // Calculate crop window overlay parameters
  const totalDuration = result?.source_duration_seconds || audioDuration || 4;
  const cropStart = result?.analysis_start_seconds ?? 0;
  const cropDuration = result?.duration_analyzed_seconds || 4;
  const cropStartPercent = totalDuration > 0 ? (cropStart / totalDuration) * 100 : 0;
  const cropWidthPercent = totalDuration > 0 ? (cropDuration / totalDuration) * 100 : 100;
  const playbackPercent = audioDuration > 0 ? (currentTime / audioDuration) * 100 : 0;

  return (
    <div className="space-y-6">
      {/* Workspace Header */}
      <div className="bg-[#191B30]/80 border border-[#A78BFA]/30 rounded-2xl p-6 backdrop-blur-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-white/10 gap-3">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-[#A78BFA]/10 border border-[#A78BFA]/30 flex items-center justify-center text-[#A78BFA]">
                <Mic className="w-5 h-5" />
              </div>
              <h3 className="text-xl font-bold tracking-tight text-[#A78BFA]">
                AUDIO FORENSIC LAB
              </h3>
            </div>
            <p className="text-xs md:text-sm text-[#94A3B8] mt-1.5">
              Synthetic-speech detection and acoustic forensics powered by PellaV2 with Wav2Vec-BERT backbone.
            </p>
          </div>

          <span className="font-mono text-xs font-bold px-3 py-1.5 rounded-lg border border-[#A78BFA]/30 bg-[#A78BFA]/10 text-[#A78BFA] self-start sm:self-auto">
            FORMATS: WAV, MP3, M4A, FLAC, OGG, AAC
          </span>
        </div>

        {/* Upload & Action Area */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
          <div className="lg:col-span-2">
            <UploadZone
              onFileSelected={handleFileSelected}
              acceptedFormats={['.WAV', '.MP3', '.M4A', '.FLAC', '.OGG', '.AAC']}
              acceptedMimeTypes="audio/*"
              mediaLabel="audio recording"
              selectedFile={selectedFile}
              onClearFile={clearFile}
              accentColor="violet"
            />
          </div>

          {/* Action Card */}
          <div className="flex flex-col justify-between bg-[#121424] border border-white/5 rounded-2xl p-5">
            <div>
              <div className="font-mono text-xs text-[#A78BFA] font-bold uppercase tracking-wider mb-2">
                Screening Protocol
              </div>
              <p className="text-xs text-[#94A3B8] leading-relaxed mb-4">
                Evaluate media against PellaV2 fine-tuned neural representations for synthetic speech artifacts.
              </p>

              <div className="p-3 rounded-xl bg-[#10111F] border border-white/5 font-mono text-[11px] text-[#94A3B8] space-y-1.5">
                <div className="flex justify-between">
                  <span>SAMPLE RATE:</span>
                  <span className="text-white">16,000 Hz</span>
                </div>
                <div className="flex justify-between">
                  <span>CHANNELS:</span>
                  <span className="text-white">Mono (1 ch)</span>
                </div>
                <div className="flex justify-between">
                  <span>WINDOW:</span>
                  <span className="text-[#A78BFA]">4s Center Crop</span>
                </div>
                <div className="flex justify-between">
                  <span>SCORE CUTOFF:</span>
                  <span className="text-[#A78BFA]">≥ 0.500</span>
                </div>
              </div>
            </div>

            <button
              disabled={!selectedFile || isScanning || !audioDetectorAvailable}
              onClick={runAnalysis}
              className="mt-4 w-full py-3 px-4 rounded-xl font-bold text-xs tracking-wider uppercase text-white bg-gradient-to-r from-[#7928CA] via-[#A78BFA] to-[#00D2FF] border border-white/20 hover:shadow-[0_0_25px_rgba(167,139,250,0.4)] transition-all disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer flex items-center justify-center gap-2"
            >
              {isScanning ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" />
                  <span>ANALYZING WITH PELLAV2...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>ANALYSE AUDIO ⚡</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Backend / Model Availability Warning */}
        {!audioDetectorAvailable && (
          <div
            className="mt-5 p-4 rounded-xl bg-[#A78BFA]/10 border border-[#A78BFA]/40 text-xs text-[#C4B5FD]"
            role="status"
          >
            <strong className="text-white">Audio analysis unavailable:</strong>{' '}
            {backendStatus.audioUnavailableReason ||
              'PellaV2 was not initialized; no audio verdict will be produced.'}
          </div>
        )}

        {/* Error Alert */}
        {errorMessage && (
          <div className="mt-5 p-4 rounded-xl bg-[#FF3366]/15 border border-[#FF3366]/40 flex items-start gap-3 text-xs text-[#FF75D8]">
            <AlertCircle className="w-5 h-5 text-[#FF3366] shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">Execution Notice:</strong> {errorMessage}
            </div>
          </div>
        )}

        {/* Waveform & Audio Player Section */}
        {previewUrl && (
          <div className="mt-6 pt-6 border-t border-white/10 space-y-3">
            <div className="flex items-center justify-between font-mono text-xs text-[#A78BFA]">
              <span>AUDIO SIGNAL INSPECTION</span>
              <span className="text-[#94A3B8]">{selectedFile?.name}</span>
            </div>

            {/* Preview state: converting indicator */}
            {previewState === 'converting' && (
              <div className="flex items-center gap-2 text-xs text-[#A78BFA] bg-[#A78BFA]/10 border border-[#A78BFA]/30 rounded-xl px-3 py-2">
                <RefreshCw className="w-3.5 h-3.5 animate-spin shrink-0" />
                <span>Converting to browser-compatible format for preview — forensic analysis will use the original file.</span>
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
                <Volume2 className="w-3.5 h-3.5 shrink-0" />
                <span>Preview converted to MP3 for browser playback. Original file submitted for PellaV2 forensic analysis.</span>
              </div>
            )}

            <div className="rounded-2xl border border-[#A78BFA]/25 bg-[#0A0B14]/80 p-4 md:p-5 space-y-4">
              {/* Waveform Viewport with Interactive Scrubbing */}
              <div
                className="relative h-28 w-full rounded-xl bg-[#07080F] border border-white/10 p-2 cursor-pointer select-none overflow-hidden"
                onClick={handleWaveformClick}
                title="Click anywhere to seek"
              >
                {/* 4-second Center Crop Evaluated Window Overlay */}
                {result && (
                  <div
                    className="absolute top-0 bottom-0 bg-[#A78BFA]/15 border-x border-[#A78BFA]/70 pointer-events-none transition-all duration-300"
                    style={{
                      left: `${cropStartPercent}%`,
                      width: `${cropWidthPercent}%`,
                    }}
                  >
                    <div className="absolute top-1 left-1.5 font-mono text-[9px] uppercase tracking-wider text-[#A78BFA] bg-[#10111F]/80 px-1.5 py-0.5 rounded border border-[#A78BFA]/30">
                      Evaluated Window ({cropDuration.toFixed(1)}s)
                    </div>
                  </div>
                )}

                {/* Waveform Bars */}
                <div className="h-full w-full flex items-center justify-between gap-[2px]">
                  {waveformPeaks.map((peak, index) => {
                    const barPercent = (index / waveformPeaks.length) * 100;
                    const isInCrop =
                      result &&
                      barPercent >= cropStartPercent &&
                      barPercent <= cropStartPercent + cropWidthPercent;
                    const isPlayed = barPercent <= playbackPercent;

                    let barColor = 'bg-[#64748B]/40';
                    if (isPlayed) {
                      barColor = isInCrop ? 'bg-[#FF75D8]' : 'bg-[#62E8FF]';
                    } else if (isInCrop) {
                      barColor = 'bg-[#A78BFA]';
                    }

                    return (
                      <div
                        key={index}
                        className={`w-full rounded-full transition-all duration-100 ${barColor}`}
                        style={{
                          height: `${Math.round(peak * 100)}%`,
                          minHeight: '4px',
                        }}
                      />
                    );
                  })}
                </div>

                {/* Playhead Indicator Line */}
                <div
                  className="absolute top-0 bottom-0 w-[2px] bg-white shadow-[0_0_8px_#fff] pointer-events-none"
                  style={{ left: `${playbackPercent}%` }}
                />
              </div>

              {/* Player Controls Bar */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={togglePlay}
                    className="w-9 h-9 rounded-xl bg-[#A78BFA]/20 border border-[#A78BFA]/40 hover:bg-[#A78BFA]/30 text-[#A78BFA] hover:text-white flex items-center justify-center transition-colors cursor-pointer"
                    aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                  </button>

                  <div className="font-mono text-xs text-[#94A3B8]">
                    <span className="text-white font-semibold">{formatSeconds(currentTime)}</span>
                    <span> / </span>
                    <span>{formatSeconds(audioDuration)}</span>
                  </div>
                </div>

                {/* Temporal Evaluation Pill */}
                {result && (
                  <div className="flex items-center gap-2 font-mono text-[10px] text-[#A78BFA] bg-[#A78BFA]/10 border border-[#A78BFA]/25 px-3 py-1.5 rounded-lg self-start sm:self-auto">
                    <Activity className="w-3.5 h-3.5" />
                    <span>
                      Window: {cropStart.toFixed(2)}s – {(cropStart + cropDuration).toFixed(2)}s (
                      {result.analysis_region === 'center_crop' ? 'Center Crop' : 'Full Short Clip'})
                    </span>
                  </div>
                )}
              </div>

              {/* Hidden native audio element */}
              <audio
                ref={audioRef}
                src={previewUrl ?? undefined}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onError={handleAudioError}
                onEnded={() => setIsPlaying(false)}
                className="hidden"
              />
            </div>
          </div>
        )}

        {/* Scanning Animation */}
        {isScanning && (
          <ScanningOverlay
            currentPhase={scanPhase}
            progressPercent={scanProgress}
            mediaType="audio"
            modelLabel="PellaV2 · 16kHz Mono · 4s Center Crop"
          />
        )}

        {/* RESULTS SECTION */}
        {result && (
          <div className="mt-8 space-y-4 animate-in fade-in duration-500">
            {/* 1. Main Verdict Banner (Primary Visual Focus) */}
            <VerdictBanner
              verdict={result.verdict}
              score={result.score}
              scoreCutoff={0.5}
              scoreLabel="Uncalibrated Detector Score"
              scoreScale="fraction"
              sourceLabel="PellaV2 Detector"
              latencyMs={result.inferenceDurationMs}
              thresholdExplanation="Threshold Rule: Score at or above 0.500 → AI-generated; below 0.500 → Likely Real. (Sigmoid-transformed, uncalibrated score)"
              subtitle={
                result.verdict === 'AI-generated'
                  ? `Uncalibrated detector score crossed the 0.500 decision threshold, flagging synthetic-speech indicators in the ${
                      result.analysis_region === 'center_crop' ? '4-second center window' : 'recording'
                    }.`
                  : `Uncalibrated detector score remained below the 0.500 decision threshold, indicating likely authentic speech in the ${
                      result.analysis_region === 'center_crop' ? '4-second center window' : 'recording'
                    }.`
              }
            />

            {/* 2. Compact Telemetry Strip */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-[#121424] border border-white/5 rounded-xl p-3">
                <div className="font-mono text-[10px] text-[#8290A8] uppercase tracking-wider">
                  Analyzed Window
                </div>
                <div className="font-mono text-base font-bold text-white mt-0.5">
                  {result.duration_analyzed_seconds.toFixed(1)}s
                  {result.source_duration_seconds !== undefined && (
                    <span className="text-xs text-[#8290A8] font-normal ml-1">
                      / {result.source_duration_seconds.toFixed(1)}s
                    </span>
                  )}
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3">
                <div className="font-mono text-[10px] text-[#8290A8] uppercase tracking-wider">
                  Window Region
                </div>
                <div className="font-mono text-base font-bold text-[#A78BFA] mt-0.5">
                  {result.analysis_region === 'center_crop' ? 'Center Crop' : 'Full Short Clip'}
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3">
                <div className="font-mono text-[10px] text-[#8290A8] uppercase tracking-wider">
                  Preprocessing
                </div>
                <div className="font-mono text-base font-bold text-[#62E8FF] mt-0.5">
                  16 kHz Mono
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3">
                <div className="font-mono text-[10px] text-[#8290A8] uppercase tracking-wider">
                  Architecture
                </div>
                <div className="font-mono text-base font-bold text-[#10B981] mt-0.5">
                  Wav2Vec-BERT
                </div>
              </div>
            </div>

            {/* 3. Probability Calibration (Kept separate, explicit, single short notice) */}
            <section
              className="rounded-xl border border-white/10 bg-[#121424]/90 p-4"
              aria-label="Probability calibration"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs uppercase tracking-wider text-[#A78BFA] font-semibold">
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
                Calibrated probability is currently unavailable. TruthLens reports the raw uncalibrated detector score (sigmoid of model output); calibrated probabilities are withheld until an independent calibrator is validated.
              </p>
            </section>

            {/* 4. Expandable Audio Methodology & Technical Caveats */}
            <MethodologyDetails title="Acoustic Methodology & Operational Caveats">
              <p>
                <strong className="text-white">PellaV2 Detector Architecture:</strong> Evaluates audio with a Wav2Vec2-BERT backbone fine-tuned to discriminate natural speech from speech synthesis, voice cloning, and neural vocoder artifacts.
              </p>
              <p>
                <strong className="text-white">Preprocessing:</strong> FFmpeg decodes and resamples the audio stream into 16,000 Hz single-channel 16-bit PCM. If the recording exceeds 4.0 seconds, the central 4-second crop is extracted for inference. Clips shorter than 4 seconds are zero-padded to match the model input shape.
              </p>
              <p>
                <strong className="text-white">Operational Boundaries:</strong> {result.note}
              </p>
              <p className="text-[11px] text-[#8290A8]">
                Acoustic compression, re-encoding, low-bitrate codecs, or background noise can influence detector representations. Replayed or re-recorded synthetic speech may not exhibit original vocoder signatures.
              </p>
            </MethodologyDetails>
          </div>
        )}
      </div>
    </div>
  );
};
