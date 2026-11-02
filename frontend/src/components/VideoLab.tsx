import React, { useState } from 'react';
import {
  Video as VideoIcon,
  Zap,
  RotateCw,
  AlertCircle,
  Film,
  Volume2,
  Clock,
  Layers,
  AlertTriangle,
  Info,
} from 'lucide-react';
import { UploadZone } from './UploadZone';
import { ScanningOverlay } from './ScanningOverlay';
import { VideoAnalysisResult, HistoryItem, BackendStatus } from '../types/forensics';
import { analyzeVideo } from '../services/api';

interface VideoLabProps {
  onScanCompleted: (item: HistoryItem) => void;
  backendStatus: BackendStatus;
}

export const VideoLab: React.FC<VideoLabProps> = ({ onScanCompleted, backendStatus }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanPhase, setScanPhase] = useState('');
  const [scanProgress, setScanProgress] = useState(0);
  const [result, setResult] = useState<VideoAnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setErrorMessage(null);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
  };

  const clearFile = () => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setSelectedFile(null);
    setVideoUrl(null);
    setResult(null);
    setErrorMessage(null);
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;

    if (!backendStatus.connected) {
      setErrorMessage(
        "The Python detector backend is offline. From the backend folder, run .\\.venv\\Scripts\\python.exe -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000."
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
      setScanPhase('Evaluating keyframes with ViT and audio with PellaV2...');
    }, 1700);

    try {
      const analysis = await analyzeVideo(selectedFile);
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      setScanProgress(100);
      setIsScanning(false);
      setResult(analysis);

      // Record in session history
      const hasAiFrames = analysis.ai_flagged_frames > 0;
      const hasAiAudio = analysis.audio_analysis?.verdict === 'AI-generated';
      const verdictSummary = hasAiFrames || hasAiAudio
        ? 'AI-FLAGGED (analyzed samples)'
        : 'No indicators in analyzed samples';

      const historyItem: HistoryItem = {
        id: `vid-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        fileName: selectedFile.name,
        mediaType: 'video',
        fileSizeFormatted: `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB`,
        verdictDisplay: verdictSummary,
        scoreDisplay: `Mean sampled-frame score: ${(analysis.average_frame_score * 100).toFixed(1)}% • Audio model score: ${
          analysis.audio_analysis ? `${(analysis.audio_analysis.score * 100).toFixed(1)}%` : 'not analyzed'
        }`,
        details: `${analysis.frames_analyzed} frames • ${analysis.ai_flagged_frames} flagged • ${analysis.duration_seconds}s`,
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

  // Check if modalities disagree
  const visualFlagged = (result?.ai_flagged_frames || 0) > 0;
  const audioFlagged = result?.audio_analysis?.verdict === 'AI-generated';
  const modalitiesDisagree = result && result.audio_analysis !== null && visualFlagged !== audioFlagged;

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
              Dual-pipeline evaluation: samples up to eight uniform frames evaluated via ViT, and extracts the audio track evaluated separately via PellaV2.
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
                Frames and audio are evaluated independently. Modality scores are never combined into a single fabricated score.
              </p>

              <div className="p-3 rounded-xl bg-[#10111F] border border-white/5 font-mono text-[11px] text-[#94A3B8] space-y-1">
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
                  <span className="text-[#FF75D8]">Demuxed &amp; Separate</span>
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
        {videoUrl && (
          <div className="mt-6 pt-6 border-t border-white/10">
            <div className="flex items-center justify-between mb-3 font-mono text-xs text-[#FF75D8]">
              <span>LOADED VIDEO CONTAINER</span>
              <span className="text-[#94A3B8]">{selectedFile?.name}</span>
            </div>
            <div className="relative rounded-2xl bg-[#0E0F1C] border border-[#FF75D8]/30 p-3 overflow-hidden max-w-2xl mx-auto shadow-xl">
              <video src={videoUrl} controls className="w-full rounded-xl max-h-72 object-contain bg-black" />
            </div>
          </div>
        )}

        {/* Scanning Animation */}
        {isScanning && (
          <ScanningOverlay
            currentPhase={scanPhase}
            progressPercent={scanProgress}
            mediaType="video"
            modelLabel="ViT-Small/16 (Frames) + PellaV2 (Demuxed Audio)"
          />
        )}

        {/* RESULTS SECTION */}
        {result && (
          <div className="mt-8 space-y-6 animate-in fade-in duration-500">
            {/* Visual Disagreement Alert Callout if modalities disagree */}
            {modalitiesDisagree && (
              <div className="p-4 rounded-2xl bg-[#FF75D8]/10 border border-[#FF75D8]/40 flex items-start gap-3 text-xs text-[#F5F5FF]">
                <AlertTriangle className="w-5 h-5 text-[#FF75D8] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[#FF75D8] uppercase tracking-wide">
                    Modality Disagreement Notice:
                  </strong>{' '}
                  Visual frame analysis and audio analysis yielded different verdicts. Visual frames flagged{' '}
                  <span className="font-bold text-[#FF75D8]">{result.ai_flagged_frames} frame(s)</span> as AI, while
                  the audio track was flagged as{' '}
                  <span className="font-bold text-[#62E8FF]">{result.audio_analysis?.verdict}</span>. Neither result
                  alone proves whether the complete video is authentic.
                </div>
              </div>
            )}

            {/* Macro Summary Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">VIDEO DURATION</div>
                <div className="font-mono text-xl font-bold text-[#62E8FF] mt-0.5">
                  {result.duration_seconds.toFixed(2)}s
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">FFprobe measured runtime</div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">FRAMES ANALYSED</div>
                <div className="font-mono text-xl font-bold text-[#A78BFA] mt-0.5">
                  {result.frames_analyzed} Frames
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">
                  {result.frames_requested ?? result.frames_analyzed} requested; samples do not cover every moment
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">FRAMES FLAGGED AS AI</div>
                <div className="font-mono text-xl font-bold text-[#FF75D8] mt-0.5">
                  {result.ai_flagged_frames} / {result.frames_analyzed}
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">Logit &ge; 1.359375</div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">AVERAGE FRAME SCORE</div>
                <div className="font-mono text-xl font-bold text-white mt-0.5">
                  {(result.average_frame_score * 100).toFixed(2)}%
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">{result.frame_score_note || 'Mean uncalibrated score; not a video probability'}</div>
              </div>
            </div>

            {/* CINEMATIC FRAME-ANALYSIS TIMELINE */}
            <div className="rounded-2xl bg-[#14162B] border border-white/10 p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Film className="w-4 h-4 text-[#FF75D8]" />
                  <h4 className="font-bold text-sm text-[#F5F5FF] tracking-tight">
                    VIDEO FRAME ANALYSIS TIMELINE ({result.frames_analyzed} SAMPLES)
                  </h4>
                </div>
                <span className="font-mono text-[10px] text-[#94A3B8] bg-white/5 px-2.5 py-0.5 rounded border border-white/10">
                  ViT-Small/16 Per-Frame
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                {result.frame_results.map((frame, idx) => {
                  const isFrameAi = frame.verdict === 'AI-generated';
                  return (
                    <div
                      key={idx}
                      className={`p-3.5 rounded-xl border transition-all ${
                        isFrameAi
                          ? 'bg-[#FF3366]/15 border-[#FF3366]/40 text-white'
                          : 'bg-[#10111F] border-white/10 text-white'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-mono text-[11px] text-[#94A3B8]">
                          FRAME {idx + 1} &bull; {frame.timestamp_seconds.toFixed(2)}s
                        </span>
                        <span
                          className={`font-mono text-[10px] font-bold px-1.5 py-0.5 rounded uppercase ${
                            isFrameAi
                              ? 'bg-[#FF75D8]/20 text-[#FF75D8] border border-[#FF75D8]/40'
                              : 'bg-[#62E8FF]/20 text-[#62E8FF] border border-[#62E8FF]/40'
                          }`}
                        >
                          {frame.verdict}
                        </span>
                      </div>

                      <div className="space-y-1 font-mono text-xs">
                        <div className="flex justify-between items-center">
                          <span className="text-[#64748B]">MODEL SCORE:</span>
                          <span className="font-bold">{(frame.score * 100).toFixed(2)}%</span>
                        </div>
                        <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
                          <div
                            style={{ width: `${frame.score * 100}%` }}
                            className={`h-full ${isFrameAi ? 'bg-[#FF75D8]' : 'bg-[#62E8FF]'}`}
                          />
                        </div>
                        <div className="flex justify-between items-center pt-1 text-[11px]">
                          <span className="text-[#64748B]">RAW LOGIT:</span>
                          <span className="text-[#A78BFA]">{frame.logit.toFixed(3)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* SEPARATE AUDIO ANALYSIS PANEL */}
            <div className="rounded-2xl bg-[#14162B] border border-[#A78BFA]/30 p-5">
              <div className="flex items-center justify-between mb-3 pb-3 border-b border-white/10">
                <div className="flex items-center gap-2">
                  <Volume2 className="w-4 h-4 text-[#A78BFA]" />
                  <h4 className="font-bold text-sm text-[#A78BFA] uppercase tracking-wide">
                    Video Audio Track Analysis (Separate Modality)
                  </h4>
                </div>

                {result.audio_analysis ? (
                  <span
                    className={`font-mono text-xs font-bold px-2.5 py-0.5 rounded uppercase ${
                      result.audio_analysis.verdict === 'AI-generated'
                        ? 'bg-[#FF75D8]/20 text-[#FF75D8] border border-[#FF75D8]/40'
                        : 'bg-[#62E8FF]/20 text-[#62E8FF] border border-[#62E8FF]/40'
                    }`}
                  >
                    {result.audio_analysis.verdict}
                  </span>
                ) : (
                  <span className="font-mono text-xs text-[#94A3B8]">
                    {result.audio_track_present ? 'AUDIO STREAM UNREADABLE' : 'NO AUDIO STREAM'}
                  </span>
                )}
              </div>

              {result.audio_analysis ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="bg-[#10111F] p-3 rounded-xl border border-white/5 font-mono text-xs">
                      <div className="text-[10px] text-[#64748B]">AUDIO FAKE SCORE</div>
                      <div className="text-base font-bold text-white mt-0.5">
                        {(result.audio_analysis.score * 100).toFixed(2)}%
                      </div>
                      <div className="text-[10px] text-[#64748B]">Uncalibrated model score, not probability</div>
                    </div>

                    <div className="bg-[#10111F] p-3 rounded-xl border border-white/5 font-mono text-xs">
                      <div className="text-[10px] text-[#64748B]">DURATION ANALYSED</div>
                      <div className="text-base font-bold text-[#62E8FF] mt-0.5">
                        {result.audio_analysis.duration_analyzed_seconds.toFixed(2)}s
                      </div>
                      <div className="text-[10px] text-[#64748B]">
                        {result.audio_analysis.analysis_region === 'center_crop'
                          ? `Center window starts at ${result.audio_analysis.analysis_start_seconds?.toFixed(2) ?? '0.00'}s in the full video`
                          : 'Full short audio track; padded if needed'}
                      </div>
                    </div>

                    <div className="bg-[#10111F] p-3 rounded-xl border border-white/5 font-mono text-xs">
                      <div className="text-[10px] text-[#64748B]">CONVERSION SPEC</div>
                      <div className="text-base font-bold text-[#A78BFA] mt-0.5">16,000 Hz</div>
                      <div className="text-[10px] text-[#64748B]">Mono PCM 16-bit</div>
                    </div>
                  </div>

                  <p className="text-xs text-[#94A3B8] leading-relaxed pt-1">
                    <strong className="text-white">Acoustic Note:</strong> {result.audio_analysis.note}
                  </p>
                </div>
              ) : (
                <p className="text-xs text-[#94A3B8]">
                  {result.audio_track_present
                    ? 'An audio stream is present but could not be decoded for analysis. The sampled visual frame results are shown separately.'
                    : 'No audio stream was found. The sampled visual frame results are shown separately.'}
                </p>
              )}
            </div>

            {/* Official Detector Interpretation */}
            <div className="p-4 rounded-xl bg-[#121424] border border-white/10 text-xs text-[#94A3B8] leading-relaxed">
              <strong className="text-white">Evaluation Rule:</strong> {result.interpretation}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
