import React, { useState } from 'react';
import {
  Mic,
  Zap,
  RotateCw,
  AlertCircle,
} from 'lucide-react';
import { UploadZone } from './UploadZone';
import { VerdictBanner } from './VerdictBanner';
import { ScanningOverlay } from './ScanningOverlay';
import { AudioAnalysisResult, HistoryItem, BackendStatus } from '../types/forensics';
import { analyzeAudio } from '../services/api';

interface AudioLabProps {
  onScanCompleted: (item: HistoryItem) => void;
  backendStatus: BackendStatus;
}

export const AudioLab: React.FC<AudioLabProps> = ({ onScanCompleted, backendStatus }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanPhase, setScanPhase] = useState('');
  const [scanProgress, setScanProgress] = useState(0);
  const [result, setResult] = useState<AudioAnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setErrorMessage(null);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
  };

  const clearFile = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setSelectedFile(null);
    setAudioUrl(null);
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
    setScanProgress(20);
    setScanPhase('Demuxing and converting input to 16,000 Hz mono PCM via FFmpeg...');

    const timer1 = setTimeout(() => {
      setScanProgress(55);
      setScanPhase('Selecting the full short clip or the 4-second center window...');
    }, 450);

    const timer2 = setTimeout(() => {
      setScanProgress(85);
      setScanPhase('Running PellaV2 Wav2Vec-BERT inference...');
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
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        fileName: selectedFile.name,
        mediaType: 'audio',
        fileSizeFormatted: `${(selectedFile.size / 1024).toFixed(1)} KB`,
        verdictDisplay: analysis.verdict,
        scoreDisplay: `${(analysis.score * 100).toFixed(2)}%`,
        details: `${analysis.duration_analyzed_seconds}s analyzed • ${analysis.analysis_region || 'audio window'} • 16kHz mono`,
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
              PellaV2 synthetic-speech screening. Converts input to 16 kHz mono PCM, analyzes up to a 4-second center crop, and pads shorter clips.
            </p>
          </div>

          <span className="font-mono text-xs font-bold px-3 py-1.5 rounded-lg border border-[#A78BFA]/30 bg-[#A78BFA]/10 text-[#A78BFA] self-start sm:self-auto">
            FORMATS: WAV, MP3, M4A, FLAC, OGG, AAC
          </span>
        </div>

        {/* Upload Zone */}
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
                Acoustic Parameters
              </div>
              <p className="text-xs text-[#94A3B8] leading-relaxed mb-4">
                FFmpeg converts the input file to 16,000 Hz 1-channel mono PCM prior to PellaV2 evaluation.
              </p>

              <div className="p-3 rounded-xl bg-[#10111F] border border-white/5 font-mono text-[11px] text-[#94A3B8] space-y-1">
                <div className="flex justify-between">
                  <span>SAMPLE RATE:</span>
                  <span className="text-white">16,000 Hz</span>
                </div>
                <div className="flex justify-between">
                  <span>CHANNELS:</span>
                  <span className="text-white">Mono (1 ch)</span>
                </div>
                <div className="flex justify-between">
                  <span>ANALYSIS WINDOW:</span>
                  <span className="text-[#A78BFA]">4s Centre Crop</span>
                </div>
              </div>
            </div>

            <button
              disabled={!selectedFile || isScanning}
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

        {/* Backend Error Alert if any */}
        {errorMessage && (
          <div className="mt-5 p-4 rounded-xl bg-[#FF3366]/15 border border-[#FF3366]/40 flex items-start gap-3 text-xs text-[#FF75D8]">
            <AlertCircle className="w-5 h-5 text-[#FF3366] shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">Execution Notice:</strong> {errorMessage}
            </div>
          </div>
        )}

        {/* Uploaded audio player */}
        {audioUrl && (
          <div className="mt-6 pt-6 border-t border-white/10">
            <div className="flex items-center justify-between mb-3 font-mono text-xs text-[#A78BFA]">
              <span>LOADED AUDIO SOURCE</span>
              <span className="text-[#94A3B8]">{selectedFile?.name}</span>
            </div>

            <div className="p-4 rounded-2xl bg-[#0E0F1C] border border-[#A78BFA]/30 space-y-4">
              <audio src={audioUrl} controls className="w-full rounded-xl bg-[#121424]" />

              <p className="text-xs text-[#94A3B8] leading-relaxed">
                The audio player is the uploaded recording. No waveform or region graphic is shown because the detector does not return time-aligned signal evidence.
              </p>
            </div>
          </div>
        )}

        {/* Scanning Animation */}
        {isScanning && (
          <ScanningOverlay
            currentPhase={scanPhase}
            progressPercent={scanProgress}
            mediaType="audio"
            modelLabel="PellaV2 (16kHz Mono • 4s Centre Crop)"
          />
        )}

        {/* RESULTS SECTION */}
        {result && (
          <div className="mt-8 space-y-5 animate-in fade-in duration-500">
            <VerdictBanner
              verdict={result.verdict}
              score={result.score}
              sourceLabel="PellaV2 Detector"
              latencyMs={result.inferenceDurationMs}
              subtitle={
                result.verdict === 'AI-generated'
                  ? `Model score crossed the 0.50 cutoff, flagging synthetic-speech indicators in the ${result.analysis_region === 'center_crop' ? '4-second center crop' : 'full short recording'}.`
                  : `Model score remained below the 0.50 cutoff, indicating likely real speech in the ${result.analysis_region === 'center_crop' ? '4-second center crop' : 'full short recording'} according to this model.`
              }
            />

            {/* Detailed Telemetry Chips */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">MODEL SCORE</div>
                <div className="font-mono text-xl font-bold text-[#A78BFA] mt-0.5">
                  {(result.score * 100).toFixed(2)}%
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">
                  Binary cutoff at 50.00%
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">ANALYSIS DURATION</div>
                <div className="font-mono text-xl font-bold text-[#62E8FF] mt-0.5">
                  {result.duration_analyzed_seconds.toFixed(2)}s / {result.source_duration_seconds?.toFixed(2) ?? '—'}s
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">
                  {result.analysis_region === 'center_crop'
                    ? `Center crop starts at ${result.analysis_start_seconds?.toFixed(2) ?? '0.00'}s`
                    : 'Full short recording; zero-padded if needed'}
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">PREPROCESSING</div>
                <div className="font-mono text-base font-bold text-[#FF75D8] mt-0.5">
                  16 kHz Mono
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">
                  FFmpeg PCM 16-bit
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">MODEL IDENTIFIER</div>
                <div className="font-mono text-xl font-bold text-[#10B981] mt-0.5">
                  PellaV2
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">
                  Wav2Vec-BERT features
                </div>
              </div>
            </div>

            {/* Verified Note */}
            <div className="p-4 rounded-xl bg-[#A78BFA]/10 border border-[#A78BFA]/25 text-xs text-[#94A3B8] leading-relaxed">
              <strong className="text-[#A78BFA]">MODEL LIMITATION NOTE:</strong> {result.note}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
